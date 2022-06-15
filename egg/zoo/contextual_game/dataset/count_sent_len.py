import json
from collections import Counter
from pathlib import Path

import clip

tokenizer = clip.simple_tokenizer.SimpleTokenizer()

data_path = Path("/private/home/rdessi/imagecode/data/")

# not including the test set since it is unlabeled and not used
with open(data_path / "train_data.json") as fd:
    train = json.load(fd)
with open(data_path / "valid_data.json") as fd:
    valid = json.load(fd)

train_and_valid = {**train, **valid}

caption_len = []
for _, captions in train_and_valid.items():
    for caption in captions.values():
        tokenized = tokenizer.encode(caption)
        caption_len.append(len(tokenized))

counter = Counter(caption_len)
# 47 captions are longer than 77
