import argparse
import logging
import json
import random
from pathlib import Path
from collections import Counter


logger = logging.getLogger(__name__)


def load_data(dataset_dir: str, output_folder: str, train_data: int = 80):
    path = Path(dataset_dir)
    path_objects, path_image_data = path / "objects.json", path / "image_data.json"
    output_folder = Path(output_folder)
    #train_objs = output_folder / "train_objects.json"
    #train_imgs = output_folder / "train_image_data.json"
    #val_objs = output_folder / "val_objects.json"
    #val_imgs = output_folder / "val_image_data.json"
    
    val_objs_easy = output_folder / "val_objects_easy.json"
    val_imgs_easy = output_folder / "val_image_data_easy.json"
    val_objs_hard = output_folder / "val_objects_hard.json"
    val_imgs_hard = output_folder / "val_image_data_hard.json"

    with open(path_image_data) as fin:
        img_data = json.load(fin)
    with open(path_objects) as fin:
        obj_data = json.load(fin)

    train_samples = int(len(img_data) * train_data / 100)
    validation_samples = len(img_data) - train_samples
    logger.info(f"Train samples: {train_samples}")
    logger.info(f"Validation samples: {validation_samples}")

    idxs = random.sample(range(len(img_data)), k=len(img_data))

    train_obj_list, train_image_data_list = [], []
    val_obj_list, val_image_data_list = [], []
    for idx in idxs:
        if idx < train_samples:
            train_obj_list.append(obj_data[idx])
            train_image_data_list.append(img_data[idx])
        else:
            val_obj_list.append(obj_data[idx])
            val_image_data_list.append(img_data[idx])
            
            
    val_obj_easy_list, val_image_data_easy_list = [], []  
    val_obj_hard_list, val_image_data_hard_list = [], []
    counter = 0
    for im_data_item, obj_data_item in zip(img_data, obj_data):
        names = [obj["names"][0] for obj in obj_data_item["objects"]]
        if names:
            if Counter(names).most_common(1)[0][1] >= 3:
                val_obj_hard_list.append(obj_data_item)
                val_image_data_hard_list.append(im_data_item)  
            else:
                val_obj_easy_list.append(obj_data_item)
                val_image_data_easy_list.append(im_data_item)
        else:
            counter += 1
    print(counter)
            

     #with open(train_objs, "w") as fout:
         #json.dump(train_obj_list, fout)
     #with open(train_imgs, "w") as fout:
         #json.dump(train_image_data_list, fout)

     #with open(val_imgs, "w") as fout:
         #json.dump(val_image_data_list, fout)
     #with open(val_objs, "w") as fout:
         #json.dump(val_obj_list, fout)
        
    with open(val_objs_easy, "w") as fout:
        json.dump(val_obj_easy_list, fout)
    with open(val_imgs_easy, "w") as fout:
        json.dump(val_image_data_easy_list, fout)
        
    with open(val_objs_hard, "w") as fout:
        json.dump(val_obj_hard_list, fout)
    with open(val_imgs_hard, "w") as fout:
        json.dump(val_image_data_hard_list, fout)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder", required=True, help="Path to a folder with VG data"
    )
    parser.add_argument(
        "--output_folder", default=None, help="Optional, if None, default is path"
    )
    parser.add_argument(
        "--train_data",
        type=int,
        default=80,
        help="Proportion of samples used in the train data",
    )
    parser.add_argument("--seed", default=111)
    opts = parser.parse_args()
    random.seed(opts.seed)
    logging.basicConfig(level=logging.INFO)
    if not opts.output_folder:
        opts.output_folder = opts.dataset_folder

    load_data(
        dataset_dir=opts.dataset_folder,
        output_folder=opts.output_folder,
        train_data=opts.train_data,
    )


if __name__ == "__main__":
    main()
