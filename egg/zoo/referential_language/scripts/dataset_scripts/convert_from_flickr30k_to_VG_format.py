import argparse
import json
import os
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--image_dir", type=str, default=None)
arg_parser.add_argument("--annotations_dir", type=str, default=None)
arg_parser.add_argument("--split_file", type=str, default=None)
arg_parser.add_argument("--output_prefix", type=str, default=None)

args = arg_parser.parse_args()


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


with open(args.split_file) as f:
    split_images = set(f.read().splitlines())

ann_names = os.listdir(args.annotations_dir)

objects_data = []
image_data = []
all_object_names = []

for ann_name in ann_names:
    if not ann_name.endswith(".xml"):
        continue

    image_id = ann_name.replace(".xml", "")
    if image_id not in split_images:
        continue
    ann_path = Path(args.annotations_dir) / ann_name
    anns = get_annotations(ann_path)
    label_coords = []
    for label, coords in anns["boxes"].items():
        label_coords.extend(product([label], coords))

    objects_list = []
    for label, coords in label_coords:
        all_object_names.append(label)
        xmin, ymin, xmax, ymax = coords
        objects_list.append(
            {
                "x": xmin,
                "y": ymin,
                "w": xmax - xmin,
                "h": ymax - ymin,
                "names": [label],
            }
        )

    objects_data.append({"image_id": int(image_id), "objects": objects_list})
    image_data.append(
        {
            "image_id": int(image_id),
            "width": anns["width"],
            "height": anns["height"],
            "url": str(Path(args.image_dir) / f"{image_id}.jpg"),
        }
    )

output_object_file_name = args.output_prefix + "_object_data.json"
with open(output_object_file_name, "w") as f:
    json.dump(objects_data, f)

output_image_file_name = args.output_prefix + "_image_data.json"
with open(output_image_file_name, "w") as f:
    json.dump(image_data, f)

label_file_name = args.output_prefix + "_classes.txt"
with open(label_file_name, "w") as f:
    for label in set(all_object_names):
        print(label, file=f)
