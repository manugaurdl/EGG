from egg.zoo.emergent_captioner.finetuning.sender import ClipCapSender
from argparse import ArgumentParser
import torch
import re
from tqdm import tqdm
import more_itertools
from transformers import GPT2Tokenizer, OPTForCausalLM
from collections import namedtuple
import textstat

def read_plaintext_files(preds):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """
    with open(preds) as fin1:
        for p in tqdm(fin1):
            pp = p.strip().split('\t')
            for p in pp:
                yield p


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("dataset_path")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device('cpu'),
    )
    args = parser.parse_args()

    model = OPTForCausalLM.from_pretrained("facebook/opt-350m").eval().to(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

    it = read_plaintext_files(args.dataset_path)

    tot_snt = 0
    tot_fkgl = 0.0
    for snt in it:
        tot_fkgl += textstat.flesch_kincaid_grade(snt)
        tot_snt += 1

    tot_fkgl /= tot_snt

    print('Avg FKGL:', tot_fkgl)


