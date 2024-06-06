from argparse import ArgumentParser
import pickle
import os
import json
import numpy as np
from pprint import pprint
from pycocoevalcap.eval import Bleu, Cider, Meteor, PTBTokenizer, Rouge, Spice


def read_plaintext_file(file):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """

    data = {}
    with open(file) as fin:
        for i, line in enumerate(fin):
            captions = line.strip().split("\t")
            data[i] = [{"caption": c} for c in captions]
    return data


def compute_nlg_metrics(predictions, gold_standard, only_cider = False, spice = True):
    tokenizer = PTBTokenizer()

    predictions = tokenizer.tokenize(predictions)
    ground_truth = tokenizer.tokenize(gold_standard)
    
    if only_cider:
        scorers = [
        (Cider(), "CIDEr"),
        ]
    
    else:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        if spice:
            scorers.append((Spice(), "SPICE"))

    summary = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ground_truth, predictions)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                summary[m] = sc
        else:
            if only_cider:
                summary[method] = scores
            else:
                summary[method] = score
    print()
    if not only_cider:
        pprint(summary)
    return summary


def get_cli_args():
    parser = ArgumentParser()
    parser.add_argument("prediction_file", help="File containing predictions.")
    parser.add_argument("gold_file", help="File containing gold captions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_evaluation():
    # args = get_cli_args()
    base_dir = "/home/manugaur/EGG/inference_preds"
    file =  "mistral_sr_both_ft_cider_SR_lamda_5e-1_lr_1e-7_final_crrclm.pkl" #blip2mistral_sr_both_ft_cider_SR_lamda_5e-1_lr_1e-7_final_crrclm.pkl"

    preds = pickle.load(open(os.path.join(base_dir, f"{file}"), "rb"))
    gt = pickle.load(open("/home/manugaur/coco/cocoid2caption.pkl" ,"rb")) 
    # gt = pickle.load(open("/home/manugaur/coco/synthetic_data/blip2mistral_preproc_5.pkl" ,"rb")) 
    gt = {cocoid : gt[cocoid]  for cocoid in preds.keys()}
    
    cocoid2idx = {int(k) : int(v) for k,v in json.load(open("/home/manugaur/nips_benchmark/misc_data/coco_test_cocoid2idx.json", "r")).items()}
    idx2cocoid = {v : k for k,v in cocoid2idx.items()}
    
    gold_standard = {}
    for idx in cocoid2idx.values():
        gold_standard[idx]= [{"caption": cap} for cap in gt[idx2cocoid[idx]]]
    
    predictions = {}
    for idx in cocoid2idx.values():
        predictions[idx]= [{"caption": preds[idx2cocoid[idx]]}]

    cider_scores = list(compute_nlg_metrics(predictions=predictions, gold_standard=gold_standard, only_cider = True)['CIDEr'])
    cocoid2cider = {idx2cocoid[idx] : cider_scores[idx] for idx in cocoid2idx.values()}
    
    with open(f"/home/manugaur/EGG/cocoid2cider/{file}", "wb") as f:
        pickle.dump(cocoid2cider, f)

    print("| CIDEr scores computed.")

if __name__ == "__main__":
    run_evaluation()