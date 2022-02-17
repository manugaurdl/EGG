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

import json
import random
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--iou_threshold", type=float, default=0.1)
arg_parser.add_argument("--min_area", type=int, default=100)
arg_parser.add_argument("--image_area_proportion", type=float, default=0.25)
arg_parser.add_argument("--min_object_count", type=int, default=3)
arg_parser.add_argument("--image_file", type=str, default=None)
arg_parser.add_argument("--objects_file", type=str, default=None)
arg_parser.add_argument("--class_file", type=str, default=None)
arg_parser.add_argument("--output_prefix", type=str, default=None)
arg_parser.add_argument("--random_seed", type=int, default=666)

args = arg_parser.parse_args()

random.seed(args.random_seed)

# file with image metadata
with open(args.image_file) as f:
    image_data = json.load(f)

# file with object lists for each image
with open(args.objects_file) as f:
    object_data = json.load(f)

# label file
class_set = set()
with open(args.class_file) as f:
    for line in f:
        F = line.strip().split(",")
        class_set.update(set(F))

object_dict = {}
for object_item in object_data:
    object_dict[object_item["image_id"]] = object_item


original_dataset_image_count = 0
original_dataset_total_object_count = 0
filtered_dataset_image_count = 0
filtered_datset_total_object_count = 0
filtered_image_list = []
filtered_object_list = []

for image_item in image_data:
    image_id = image_item["image_id"]
    object_item = object_dict[image_id]

    # area of full image
    img_area = image_item["width"] * image_item["height"]

    objects = object_item["objects"]
    random.shuffle(objects)

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
            # is any of the object names in set of accepted labels?
            # are width and length more than one pixel, and overall are above min_area?
            # is the object's area less than image_area_proportion of the whole image?
            # is the object spilling over the image?
            acceptable_label_found = False
            for name in object1["names"]:
                if name in class_set:
                    acceptable_label_found = True
                    break
            if (
                (not (acceptable_label_found))
                or (object1["w"] == 1)
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
