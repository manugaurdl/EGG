#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:42:03 2021

@author: eleonora
"""

import json
from PIL import Image
from data import VisualGenomeDataset
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from scipy import spatial
import pickle
import os
import h5py

# LOAD DATA 
path_objects = 'VG_data/objects.json'
with open(path_objects) as fin:
    object_list = json.load(fin)

path_image_data = 'VG_data/image_data.json'
with open(path_image_data) as fin:
    image_data = json.load(fin)
    
# LOAD CLASSES AND CLEAN
    
class_vocab = "VG_data/objects_vocab.txt"
obj_classes = open(class_vocab).read().split("\n")

# organize dictionary of different writings of the same word
dic_writings = {}
for i in obj_classes:
    if "," in i:
        splitted = i.split(",")
        dic_writings[splitted[0]] = splitted[0]
        dic_writings[splitted[1]] = splitted[0]

# keep just one writing version
cleaned_classes = []
for i in obj_classes:
    if "," not in i:
        cleaned_classes.append(i)
    else:  
        cleaned_classes.append(dic_writings[i.split(",")[0]]) 


# KEEP OBJECTS WITH CORRECT CLASSES

dic_name_objs = {}
for name in cleaned_classes:
    dic_name_objs[name] = []
    
for im_data, ob_data in list(zip(image_data, object_list)):
    for obj in ob_data['objects']:
        for name in obj['names']:
            if name in cleaned_classes:
                dic_name_objs[name].append([im_data['url'], obj])
            elif name in dic_writings.keys() and dic_writings[name] in cleaned_classes:
                    dic_name_objs[dic_writings[name]].append([im_data['url'], obj])
            else:
                pass




# COMPUTE FEATURES & PROTOTYPES

# Load ResNet101 and define image transformation
model = torchvision.models.resnet101(pretrained=True)
model.fc = nn.Identity()
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.4529, 0.4170, 0.3804], 
                                                     std=[0.1830, 0.1779, 0.1745])])
for param in model.parameters():
    param.requires_grad = False
model = model.eval()


def get_prototypicality(name):

    feat_list = []
    similarities = []
    for obj in dic_name_objs[name]:
        im_name = "VG_data/" + obj[0].split("/")[-2] + "/" + obj[0].split("/")[-1]
        im = pil_loader(im_name)
        img_w, img_h = im.size
        x = obj[1]['x']
        y = obj[1]['y']
        w = obj[1]['w']
        h = obj[1]['h']
        if h <= 1 or w <= 1:
            continue
        if (x+w) * (y+h) / (img_w * img_h) > 0.01:
            cropped_im = im.crop((x, y, x+w, y+h))
            cropped_t = transform(cropped_im)
            batch_t = torch.unsqueeze(cropped_t, 0)
            features = model(batch_t)
            feat_list.append(features)
    prototype = sum(feat_list)/len(feat_list)
    for obj_feat in feat_list:
        similarities.append(1-spatial.distance.cosine(prototype, obj_feat))
        
    return [prototype, similarities]


for name in cleaned_classes:
    prototype, similarities = get_prototypicality(name)
    h5f = h5py.File('VG_prototypes/' + name + ".h5", 'w')
    h5f.create_dataset(name, data = prototype)
    h5f.close()
    pickle.dump(similarities, open("VG_typicalities/" + name + ".pkl", "wb"))


