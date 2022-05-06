import json

# import clip
import spacy


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

with open("train_data.json") as fd:
    train = json.load(fd)
with open("valid_data.json") as fd:
    valid = json.load(fd)

train_and_valid = {**train, **valid}

all_sents = []
for dataset, captions in train_and_valid.items():
    for caption in captions.values():
        all_sents.append(caption)


with open("test_data_unlabeled.json") as fd:
    test = json.load(fd)


for dataset, captions in test.items():
    all_sents.extend(captions)

count_tokens = 0
types = set()
for sent in all_sents:
    doc = nlp(sent.strip())

    count = 0
    for token in doc:
        if token.pos_ not in ["SPACE"]:
            count += 1
            types.add((token.text.lower()))

    count_tokens += count

print(f"Average tokens per description: {count_tokens / len(all_sents)}")
print(f"Number of types in dataset {len(types)}")
