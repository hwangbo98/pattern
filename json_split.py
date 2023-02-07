import json
import argparse
from glob import glob
import shutil
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser(description="source & destination file path")

parser.add_argument("--json_dir", required=True, help = 'json directory path')
# parser.add_argument("--origin_img", required=True, help = 'image directory path')
# parser.add_argument("--dest_img", required=True, help = 'destination directory path')

args = parser.parse_args()

json_path = args.json_dir #"/home/hwangbo/new_api/output"
print(json_path)
# img_path =  parser.origin_img #"/mnt/hdd3/showniq/Data/lime_mix_all"
# dest_path = parser.dest_img

# json_file = [json_path + "/*.json"]
# json_list = ["/mnt/hdd3/showniq/Data/lime_mix_all/labels/202204081812071000000071.json"]
json_list = sorted(glob(json_path + "/*.json"))
# print(json_list)
total = []
for json_file in tqdm(json_list) :
    try :
        with open(json_file,"r") as fp :
            data = json.load(fp)
        
            for k in data["item_info"] :
                result = []
                img_path = json_file.replace("labels","images")
                img_path = img_path.replace("json","jpg")
                result.append(img_path) #json_file_path

                result.append(k["bounding_box"]["lt_x"]) #left top x
                result.append(k["bounding_box"]["lt_y"]) #left top y
                result.append(k["bounding_box"]["rb_x"]) #right bottom x
                result.append(k["bounding_box"]["rb_y"]) #right bottom y
                result.append(k["item_id"].split(":")[-2]) #color
                result.append(k["item_id"].split(":")[-1]) #pattern
                # print(f'file_name : {img_path}')
                # print(k["item_id"].split(":")[-2]) # pattern
                # print(k["item_id"].split(":")[-1]) # color
                # print(k["bounding_box"]["lt_x"]) #left top x
                # print(k["bounding_box"]["lt_y"]) #left top y
                # print(k["bounding_box"]["rb_x"]) #right bottom x
                # print(k["bounding_box"]["rb_y"]) #right bottom y
                total.append(result)
    except: 
        print(json_file)

y = np.zeros(len(total))

X_train, X_temp, y_train, y_temp = train_test_split(total, y, test_size = 0.3, random_state = 42)
X_valid, X_test, _, _ = train_test_split(X_temp, y_temp, test_size = 0.66, random_state = 42)  

with open(os.path.join('CP_train.pickle'), 'wb') as f:
   pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join('CP_valid.pickle'), 'wb') as f:
   pickle.dump(X_valid, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join('CP_test.pickle'), 'wb') as f:
   pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done.")

# print(total)
