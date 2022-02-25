#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:07:41 2021

@author: eleonora
"""
import pickle
import torch
from PIL import Image
from data import VisualGenomeDataset
from torchvision.transforms.functional import crop
from torchvision import transforms as T

# LOAD DATA 

objects_dir = 'VG_data/'
metadata_dir = 'VG_data/'
dataset = VisualGenomeDataset(objects_dir, metadata_dir, "train")

transform = T.Compose([T.Resize(64), T.ToTensor()])

means = []
stds = []
for item in dataset:
    image = item[0]
    img_w, img_h = image.size
    for bbox in item[2]['bboxes']:
        x = bbox.cpu().detach().numpy()[0]
        y = bbox.cpu().detach().numpy()[1]
        h = bbox.cpu().detach().numpy()[2]
        w = bbox.cpu().detach().numpy()[3]        
        if h <= 1 or w <= 1:
            continue
        if (x+w) * (y+h) / (img_w * img_h) > 0.01:
            object = image.crop((x, y, x+w, y+h))
            object_t = transform(object)
            means.append(torch.mean(object_t, dim = (1,2)))
            stds.append(torch.std(object_t, dim = (1,2)))
              


#pickle.dump(stds, open("stds_train.pkl", "wb"))
#pickle.dump(means, open("means_train.pkl", "wb"))

#mean = torch.mean(torch.tensor(means))
#std = torch.mean(torch.tensor(stds))

print(len(means))
print(len(stds))

num1_m = sum([i[0] for i in means])/len(means)
num2_m = sum([i[1] for i in means])/len(means)
num3_m = sum([i[2] for i in means])/len(means)

num1_s = sum([i[0] for i in stds])/len(stds)
num2_s = sum([i[1] for i in stds])/len(stds)
num3_s = sum([i[2] for i in stds])/len(stds)



print("mean:", num1_m, num2_m, num3_m)
print("std:", num1_s, num2_s, num3_s)
