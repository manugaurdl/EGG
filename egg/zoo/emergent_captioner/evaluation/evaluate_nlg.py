from argparse import ArgumentParser

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
    args = get_cli_args()

    predictions = read_plaintext_file(args.prediction_file)
    gold_standard = read_plaintext_file(args.gold_file)

    compute_nlg_metrics(predictions=predictions, gold_standard=gold_standard)


if __name__ == "__main__":
    run_evaluation()