from __future__ import print_function, division

import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
import glob
from tqdm import tqdm
import argparse
import pickle

meanRGB = []
stdRGB = []

# print(len(glob.glob('/home/jewoo/showniq_croped/labeling/images/human/*/*/*/*.jpg')))

# def pk_path(root_dir, type) :
#     return pk_real_path = root_dir + '/CP_' + type + '.pickle'

# parser = argparse.ArgumentParser(description="source & destination file path")

# parser.add_argument("--pk_dir", required=True, help = 'json directory path')
# # parser.add_argument("--origin_img", required=True, help = 'image directory path')
# # parser.add_argument("--dest_img", required=True, help = 'destination directory path')

# args = parser.parse_args()

# pk_dirpath = args.pk_dir #"/home/hwangbo/new_api/output"
# train_path = pk_path(pk_dirpath,'train')
# test_path = pk_path(pk_dirpath,'valid')
# valid_path = pk_path(pk_dirpath,'test')

with open("data.pickle", "rb") as f :
    data = pickle.load(f)

for idx, image in tqdm((enumerate (data))):
    im = Image.open(image[0]).convert('RGB')
    area = (image[1],image[2],image[3],image[4])
    im = im.crop((area))
    # print(im)
    im = transforms.Resize((225,225))(im)
    im = transforms.ToTensor()(im)
    # print(im.size())
    meanRGB.append(np.mean(im.numpy(), axis=(1,2)))
    stdRGB.append(np.std(im.numpy(), axis=(1,2)))

# print(meanRGB)

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print("Mean R/G/B: ", meanR, meanG, meanB)
print("Std R/G/B: ", stdR, stdG, stdB)