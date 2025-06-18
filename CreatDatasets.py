import argparse
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from utils.utils2 import *
from multiprocessing.pool import Pool
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='dataHER2', help="datasets")
    parser.add_argument('--image_dir', type=str, default='image')
    parser.add_argument('--tumortype', type=str, default='dataBRCA1')
    parser.add_argument('--nuc_seg_dir', type=str, default='segment')
    parser.add_argument('--splitlist', type=str, default='splitlist.csv') 
    parser.add_argument('--piecenumber', type=int, default='30') 
    parser.add_argument('--patchsize', type=int, default='1000') 
    parser.add_argument('--allpatch', type=str, default='select') 
    parser.add_argument('--mode', type=str, default='train') 
    parser.add_argument('--basenamelen', type=int, default='12',help='length of basename in spllist.csv') 
    opt = parser.parse_known_args()[0]
    return opt
opt = parse_args()
def generate_data(opt, pat2img, split_df, trainortest):
    pool = Pool(processes=1)
    results = pool.starmap_async(getCellData, [(opt, pat_name, pat2img) for pat_name in split_df.iloc[:, 0]])
    pool.close()
    pool.join()
    results.wait()
    outputs = results.get()
    sample_names, img_names, nuc_patches, nuc_patches_pos, nuc_patches_no = [], [], [], [], []
    for (
        sample_name_pool,
        img_name_pool,
        nuc_patch_pool,
        nuc_patch_pos_pool,
        nuc_patch_no_pool,
    ) in outputs:
        sample_names.append(sample_name_pool)
        img_names.append(img_name_pool)
        nuc_patches.append(nuc_patch_pool)
        nuc_patches_pos.append(nuc_patch_pos_pool)
        nuc_patches_no.append(nuc_patch_no_pool)

    data = {
        "x_samplename": sample_names,
        "x_imgname": img_names,
        "x_nucpatch": nuc_patches,
        "x_nucpatch_pos": nuc_patches_pos,
        "x_nucpatch_no": nuc_patches_no,
    }
    output_file = f'./data/{opt.datadir}/{trainortest}data.pkl'
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Generated data saved to: {output_file}")
    return output_file
def merge_data(input_file, output_file,tumortype):
    merged_data = {"x_samplename": [], "x_tumor": [], "x_nucpatch": [], "x_imgname": [], "x_nucpatch_pos": []}

    with open(input_file, "rb") as f:
        print("Processing: " + input_file + "\n")
        data = pickle.load(f)

        for i in range(len(data["x_samplename"])):
            for j in range(len(data["x_imgname"][i])):
                cellrankl = np.sum(data["x_nucpatch_no"][i][0][:j])
                cellrankh = np.sum(data["x_nucpatch_no"][i][0][:(j + 1)])
                if 0 <= cellrankl < cellrankh <= len(data["x_nucpatch"][i][0]):
                    merged_data["x_samplename"].append(data["x_samplename"][i])
                    merged_data["x_imgname"].append(data["x_imgname"][i][j])
                    merged_data["x_nucpatch"].append(data["x_nucpatch"][i][0][cellrankl:cellrankh])
                    merged_data["x_nucpatch_pos"].append(data["x_nucpatch_pos"][i][0][cellrankl:cellrankh])
                    merged_data["x_tumor"].append(tumortype)

    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)
    print(f"Merged data saved to: {output_file}")

img_fnames = os.listdir(os.path.join("./data/",opt.datadir, opt.image_dir))
pat2img = {}
for pat, img_fname in zip([img_fname[:int(opt.basenamelen)] for img_fname in img_fnames], img_fnames):
    if pat not in pat2img.keys(): pat2img[pat] = []
    pat2img[pat].append(img_fname)
if opt.mode=="train":
    CVf_split = pd.read_csv("./data/"+opt.datadir+'/splitlist.csv',header=None)
    CV_train=CVf_split[CVf_split.iloc[:,1] == 'train']
    CV_test=CVf_split[CVf_split.iloc[:,1] == 'test']
    tumortype=opt.tumortype
    train_input_file = generate_data(opt, pat2img, CV_train, "train")
    train_output_file = f"./data/{opt.datadir}/traindata_tmp.pkl"
    test_input_file = generate_data(opt, pat2img, CV_test, "test")
    test_output_file = f"./data/{opt.datadir}/testdata_tmp.pkl"
else:
    CV_all = pd.DataFrame({0: list(pat2img.keys())})  
    test_input_file = generate_data(opt, pat2img, CV_all, "test")
    test_output_file = f"./data/{opt.datadir}/testdata.pkl"
    merge_data(test_input_file, test_output_file, opt.tumortype)
