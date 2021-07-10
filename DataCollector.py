from os import makedirs
import os.path as path

import time

import urllib.request

import numpy as np   
import pandas as pd

from argparse import ArgumentParser
import json
# Globals
data_dir    = ""
img_dir   = ""

"""
Parsing the arguments:
- directory with data
- directory with images

Example usage: python3 DataCollector.py -d "/home/gojourney/SS2021/DS/DsPJ2021-Data/data" -i "/home/gojourney/SS2021/DS/DsPJ2021-Data/images"
"""
def parse_arguments():
    global data_dir
    global img_dir

    parser = ArgumentParser(description='Parse arguments')
    parser.add_argument('-d', '--data_dir', help='path to data directory',
                        required=True)
    parser.add_argument('-i', '--img_dir', 
                        help='path to images directory',
                        required=True)

    args = parser.parse_args()
    data_dir = args.data_dir
    img_dir = args.img_dir

def download_images(csv_label_path : str, img_dst : str):

    makedirs(img_dst, exist_ok=True)
    for [iid, imageUrl] in pd.read_csv(csv_label_path, delimiter="\t")[["iid", "imageUrl"]].values:

        # Download images from url-list:
        urllib.request.urlretrieve(imageUrl, path.join(img_dst, "%s.jpg" % iid))

    return

# download_images(path.join(data_dir, "03_label.tsv"), path.join(img_dir, "03"))

def get_train_data(csv_path : str, img_dir : str):

    data = pd.read_csv(csv_path, delimiter="\t") 
    
    adata = data[["aid", "title", "text"]]

    idata = data[["iid", "article"]]
    idata["ipath"] = idata["article"].apply(lambda anr : path.join(img_dir, "%s.jpg" % anr))
    del idata["article"]

    return data[["aid", "iid"]], adata, idata

def merge_train_data(csv_paths : list, img_dirs : list):

    data = pd.DataFrame(columns=['aid', 'iid'])
    adata = pd.DataFrame(columns=['aid', 'title', 'text'])
    idata = pd.DataFrame(columns=['iid', 'ipath'])

    for csv_path, img_dir in zip(csv_paths, img_dirs):

        temp_data, temp_adata, temp_idata = get_train_data(csv_path, img_dir)

        data = data.append(temp_data, ignore_index=True)
        adata = adata.append(temp_adata, ignore_index=True)
        idata = idata.append(temp_idata, ignore_index=True)

    return data, adata, idata

def get_test_data(csv_data_path : str, csv_label_path : str, img_dir : str):

    data_text = pd.read_csv(csv_data_path, delimiter="\t")[["article", "aid", "title", "text"]]
    data_img = pd.read_csv(csv_label_path, delimiter="\t")[["iid"]]

    data_img["path"] = data_img["iid"].apply(lambda iid: path.join(img_dir, "%s.jpg" % iid))

    return data_text, data_img 
def save_image_paths(idata : pd.DataFrame):
    l=[{"image_id":item["iid"],"image_path":item["ipath"]} for item in idata]
    with open('image_paths.json', 'w') as jsonfile:
        json.dump(l, jsonfile)


if __name__ == "__main__":

    parse_arguments()

    data_01, adata_01, idata_01 = get_train_data(path.join(data_dir, "01.tsv"), path.join(img_dir, "01"))
    data_02, adata_02, idata_02 = get_train_data(path.join(data_dir, "02.tsv"), path.join(img_dir, "02"))

    data_text, data_img = get_test_data(path.join(data_dir, "03_data.tsv"), path.join(data_dir, "03_label.tsv"), path.join(img_dir, "03"))

    data, adata, idata = merge_train_data([path.join(data_dir, "01.tsv"), path.join(data_dir, "02.tsv")],
                                          [path.join(img_dir, "01"), path.join(img_dir, "02")])
    save_image_ipaths(idata)