# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

train_path = "./train_data.json"

data = []
with open(train_path) as fd:
    data = json.load(fd)

train_samples = {}
valid_samples = {}

for img_dir, sents in data.items():
    if len(sents) == 1:
        valid_samples[img_dir] = sents
    else:
        train_samples[img_dir] = sents

with open("./new_train_data.json", "w") as f:
    json.dump(train_samples, f, indent=2)

with open("./new_valid_data.json", "w") as f:
    json.dump(valid_samples, f, indent=2)
