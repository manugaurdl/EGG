# usage examples:

# using default values of filtering parameters:

# python ./filter_and_visualize.py \
#        --image_file image_abc.json \
#        --objects_file objects_abc.json \
#        --class_file classes_1600.txt\
#        --output_prefix filtered

# note that output files will be called filtered_image_data.json and filtered_object_data.json, respectively

# changing some filtering parameters:

# python ./filter_and_visualize.py \
#        --image_file image_abc.json \
#        --objects_file objects_abc.json \
#        --class_file classes_1600.txt \
#        --output_prefix filtered \
#        --iou_threshold 0.3 \
#        --min_area 250

import argparse
import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import product

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--iou_threshold", type=float, default=0.1)
arg_parser.add_argument("--min_area", type=int, default=100)
arg_parser.add_argument("--image_area_proportion", type=float, default=0.25)
arg_parser.add_argument("--min_object_count", type=int, default=3)
arg_parser.add_argument("--objects_dir", type=str, default=None)
arg_parser.add_argument("--split_file", type=str, default=None)
arg_parser.add_argument("--output_prefix", type=str, default=None)
arg_parser.add_argument("--random_seed", type=int, default=666)

args = arg_parser.parse_args()

random.seed(args.random_seed)


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info


metadata_dir = Path(args.objects_dir)
split_file = Path(args.split_file)

original_dataset_image_count = 0
original_dataset_total_object_count = 0
filtered_dataset_image_count = 0
filtered_datset_total_object_count = 0
filtered_image_list = []
filtered_object_list = []

ann_paths = glob.iglob(f"{os.path.expanduser(metadata_dir)}/*xml")
for ann_path in ann_paths:
    image_id = Path(ann_path).stem

    anns = get_annotations(ann_path)

    boxes = []
    for label, objs in anns["boxes"].items():
        boxes.extend(product([label], objs))
    if len(boxes) < 3:
        continue
    random.shuffle(boxes)
    anns["boxes"] = boxes


obj_area = 0.0
total_objs = 0
for image_item in image_data:

    # area of full image
    img_area = anns["width"] * anns["height"]

    objects = boxes

    blacklist = set()
    for i in range(len(objects)):
        if not (i in blacklist):
            object1 = objects[i]
            area1 = object1["w"] * object1["h"]
            x11 = object1["x"]
            y11 = object1["y"]
            x12 = x11 + object1["w"]
            y12 = y11 + object1["h"]
            # general check of object sanity
            # are width and length more than one pixel, and overall are above min_area?
            # is the object's area less than image_area_proportion of the whole image?
            # is the object spilling over the image?
            if (
                (object1["w"] == 1)
                or (object1["h"] == 1)
                or (area1 < args.min_area)
                or ((area1 / img_area) > args.image_area_proportion)
                or (x12 > image_item["width"])
                or (y12 > image_item["height"])
            ):
                blacklist.add(i)
            else:
                for j in range(i + 1, len(objects)):
                    if not (j in blacklist):
                        object2 = objects[j]
                        area2 = object2["w"] * object2["h"]
                        x21 = object2["x"]
                        y21 = object2["y"]
                        x22 = x21 + object2["w"]
                        y22 = y21 + object2["h"]

                        # following adapted from
                        # https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch
                        # intersection
                        inter_x1 = max(x11, x21)
                        inter_y1 = max(y11, y21)
                        inter_x2 = min(x12, x22)
                        inter_y2 = min(y12, y22)

                        # So the intersection BBox has the coordinates (inter_x1,inter_y1) (inter_x2,inter_y2)

                        # compute the width and height of the intersection bounding box
                        inter_w = max(0, inter_x2 - inter_x1)
                        inter_h = max(0, inter_y2 - inter_y1)
                        # find the intersection area
                        intersection_area = inter_w * inter_h
                        # find the union area of both the boxes
                        union_area = area1 + area2 - intersection_area
                        # compute the ratio of overlap between the computed
                        # bounding box and the bounding box in the area list
                        iou = intersection_area / union_area

                        # if IoU is above a certain threshold, add second BBox to blacklist
                        if iou > args.iou_threshold:
                            blacklist.add(j)

    whitelist = set(range(len(objects))) - blacklist
    filtered_objects = [objects[i] for i in whitelist]

    for i in whitelist:
        obj_area += objects[i]["w"] * objects[i]["h"]
        total_objs += 1

    original_dataset_image_count = original_dataset_image_count + 1
    original_dataset_total_object_count = original_dataset_total_object_count + len(
        objects
    )

    # check for minimum object count
    if len(filtered_objects) >= args.min_object_count:
        filtered_dataset_image_count = filtered_dataset_image_count + 1
        filtered_datset_total_object_count = filtered_datset_total_object_count + len(
            filtered_objects
        )

        filtered_image_list.append(image_item)
        object_dict[image_id]["objects"] = filtered_objects
        filtered_object_list.append(object_dict[image_id])

print("avg obj area", obj_area / total_objs)

print(f"number of images in original dataset: {original_dataset_image_count}")
print(f"number of images in filtered dataset: {filtered_dataset_image_count}")

original_average_object_count = round(
    original_dataset_total_object_count / (original_dataset_image_count + 0.0), 1
)
filtered_average_object_count = round(
    filtered_datset_total_object_count / (filtered_dataset_image_count + 0.0), 1
)

print(
    f"average number of objects per image in original dataset: {original_average_object_count}"
)
print(
    f"average number of objects per image in filtered dataset: {filtered_average_object_count}"
)

output_image_file_name = args.output_prefix + "_image_data.json"
with open(output_image_file_name, "w") as f:
    json.dump(filtered_image_list, f)

output_object_file_name = args.output_prefix + "_object_data.json"
with open(output_object_file_name, "w") as f:
    json.dump(filtered_object_list, f)
