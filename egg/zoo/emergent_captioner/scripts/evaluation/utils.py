from tqdm import tqdm


def read_plaintext_files(preds, gold):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """
    with open(preds) as fin1, open(gold) as fin2:
        for i, (p, gg) in tqdm(enumerate(zip(fin1, fin2))):
            p = p.strip()
            gg = gg.strip().split("\t")
            for g in gg:
                yield i, p, g
