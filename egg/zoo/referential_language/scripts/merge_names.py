#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 11:30:49 2021

@author: eleonora
"""
import json


file_path = "/Users/eleonora/OneDrive/PhD/EmeComm/VG_data/objects.json" # objects file of one name version
file_image_data12 = "/Users/eleonora/OneDrive/PhD/EmeComm/VG_data/image_data12.json" # multi names version
file_path_12 = "/Users/eleonora/OneDrive/PhD/EmeComm/VG_data/objects12.json" # multi names version


with open (file_path) as f:
    objects1 = json.load(f)
with open (file_path_12) as f:
    objects12 = json.load(f)
with open(file_image_data12) as f:
    image_data12 = json.load(f)    
    
# check how many objects have 2 names for each file
# for objects1 we should find 0 cases 
counter_objects1 = 0
for i in objects1:
    for obj in i['objects']:
        if len(obj['names']) > 1:
            counter_objects1 += 1
print(counter_objects1)
# 0


counter_objects12 = 0
for i in objects12:
    for obj in i['objects']:
        if len(obj['names']) > 1:
            counter_objects12 += 1
print(counter_objects12)
# 59478


# we take the name from OBJECTS1 whenever we have multiple names in OBJECTS12
# based on the object_id or merged_object_id
for item1,item12 in list(zip(objects1, objects12)):
    assert (item1['image_id'] == item12['image_id'])
    for idx_obj,obj12 in enumerate(item12['objects']):
        if len(obj12['names']) > 1:
            obj_id12 = obj12['object_id']          
            for obj1 in item1['objects']:
                if obj_id12 in obj1['merged_object_ids'] or obj_id12 == obj1['object_id']:
                    obj12["names"] = obj1['names']
                


# check cases that still have multiple names
more_names = []               
for item in objects12:
    for obj in item['objects']:
        if len(obj['names']) > 1:
            more_names.append(obj)
print(len(more_names))


# does their object in appear in the objects1 file? NO
ids = [i['object_id'] for i in more_names]
for i in objects1:
    for obj in i['objects']:
        if obj['object_id'] in ids:
            print(obj)
   
# does their merged_objects_id in appear in the objects1 file? NO
ids = [i['object_id'] for i in more_names]
for i in objects1:
    for obj in i['objects']:
        if "merged_obejcts_id" in obj.keys() and obj["merged_obejcts_id"] in ids:
            print(obj) 


# then we remove them
for idx,item in enumerate(objects12):
    tmp = []
    for idx_obj,obj in enumerate(item['objects']):
        if len(obj['names']) == 1:
            tmp.append(obj)
        item['objects'] = tmp
           

# check cases that still have multiple names now
more_names = []               
for item in objects12:
    for obj in item['objects']:
        if len(obj['names']) > 1:
            more_names.append(obj)
print(len(more_names))




# CREATE a VALIDATION SET based on previous

val_objects_small = "/Users/eleonora/OneDrive/PhD/EmeComm/VG_data/splits/val_objects_small.json" # validation set for which we want to clean the names
with open(val_objects_small) as f:
    val_objects_small = json.load(f)

val_image_ids = [i['image_id'] for i in val_objects_small]
len(val_image_ids)

val_idx = []
counter = 0
for idx, i in enumerate(image_data12):
    if i['image_id'] in val_image_ids:
        counter += 1
        val_idx.append(idx)
        print(counter)
        

new_val_objects = []
for idx in val_idx:  
    new_val_objects.append(objects12[idx])


more_names = []               
for item in new_val_objects:
    for obj in item['objects']:
        if len(obj['names']) > 1:
            more_names.append(obj)
print(len(more_names))

len(new_val_objects)

new_val_objects[0]

with open('/Users/eleonora/OneDrive/PhD/EmeComm/VG_data/splits/new_val_objects_small.json', 'w') as outfile:
    json.dump(new_val_objects, outfile)
    
    
    

