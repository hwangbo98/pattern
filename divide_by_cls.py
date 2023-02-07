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
                result.append(k["item_id"].split(":")[-2]) #pattern
                result.append(k["item_id"].split(":")[-1]) #pattern
                # print(f'file_name : {img_path}')
                # print(k["item_id"].split(":")[-2]) # pattern
                # print(k["item_id"].split(":")[-1]) # pattern
                # print(k["bounding_box"]["lt_x"]) #left top x
                # print(k["bounding_box"]["lt_y"]) #left top y
                # print(k["bounding_box"]["rb_x"]) #right bottom x
                # print(k["bounding_box"]["rb_y"]) #right bottom y
                total.append(result)
    except: 
        print(json_file)


for i in range(18) : #pattern 22개 
    globals()['pattern_idx_{}' .format(i+1)] = []
for contents in total :
    # idx_name = 'pattern_idx_' + contents[5]
    if contents[6] == '1' :
        pattern_idx_1.append(contents)
    elif contents[6] == '2' :
        pattern_idx_2.append(contents)
    elif contents[6] == '3' :
        pattern_idx_3.append(contents)
    elif contents[6] == '4' :
        pattern_idx_4.append(contents)
    elif contents[6] == '5' :
        pattern_idx_5.append(contents)
    elif contents[6] == '6' :
        pattern_idx_6.append(contents)
    elif contents[6] == '7' :
        pattern_idx_7.append(contents)
    elif contents[6] == '8' :
        pattern_idx_8.append(contents)
    elif contents[6] == '9' :
        pattern_idx_9.append(contents)
    elif contents[6] == '10' :
        pattern_idx_10.append(contents)
    elif contents[6] == '11' :
        pattern_idx_11.append(contents)
    elif contents[6] == '12' :
        pattern_idx_12.append(contents)
    elif contents[6] == '13' :
        pattern_idx_13.append(contents)
    elif contents[6] == '14' :
        pattern_idx_14.append(contents)
    elif contents[6] == '15' :
        pattern_idx_15.append(contents)
    elif contents[6] == '16' :
        pattern_idx_16.append(contents)
    elif contents[6] == '17' :
        pattern_idx_17.append(contents)
    elif contents[6] == '18' :
        pattern_idx_18.append(contents)
    # elif contents[6] == '19' :
    #     pattern_idx_19.append(contents)
    # elif contents[6] == '20' :
    #     pattern_idx_20.append(contents)
    # elif contents[6] == '21' :
    #     pattern_idx_21.append(contents)
    # elif contents[6] == '22' :
    #     pattern_idx_22.append(contents)
    
# print(len(pattern_idx_20))

# for col_1 in pattern_idx_1 :
#     print(f'pattern_1 = {col_1}')
with open('pk_dir/cls_1.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_1,f)
# for col_2 in pattern_idx_2 :
# print(f'pattern_2 = {col_2}')
with open('pk_dir/cls_2.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_2,f)

# for col_3 in pattern_idx_3 :
# print(f'pattern_3 = {col_3}')
with open('pk_dir/cls_3.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_3,f)

# for col_4 in pattern_idx_4 :
# print(f'pattern_4 = {col_4}')
with open('pk_dir/cls_4.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_4,f)

# for col_6 in pattern_idx_6 :
# print(f'pattern_5 = {col_5}')
with open('pk_dir/cls_5.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_5,f)

# for col_6 in pattern_idx_6 :
# print(f'pattern_6 = {col_6}')
with open('pk_dir/cls_6.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_6,f)

# for col_7 in pattern_idx_7 :
# print(f'pattern_7 = {col_7}')
with open('pk_dir/cls_7.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_7,f)

# for col_8 in pattern_idx_8 :
# print(f'pattern_8 = {col_8}')
with open('pk_dir/cls_8.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_8,f)

# for col_9 in pattern_idx_9 :
# print(f'pattern_9 = {col_9}')
with open('pk_dir/cls_9.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_9,f)

# for col_10 in pattern_idx_10 :
# print(f'pattern_10 = {col_10}')
with open('pk_dir/cls_10.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_10,f)

# for col_11 in pattern_idx_11 :
# print(f'pattern_11 = {col_11}')
with open('pk_dir/cls_11.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_11,f)

# for col_12 in pattern_idx_12 :
# print(f'pattern_12 = {col_12}')
with open('pk_dir/cls_12.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_12,f)

# for col_13 in pattern_idx_13 :
# print(f'pattern_13 = {col_13}')
with open('pk_dir/cls_13.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_13,f)

# for col_14 in pattern_idx_14 :
# print(f'pattern_14 = {col_14}')
with open('pk_dir/cls_14.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_14,f)

# for col_15 in pattern_idx_15 :
# print(f'pattern_15 = {col_15}')
with open('pk_dir/cls_15.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_15,f)

# for col_16 in pattern_idx_16 :
# print(f'pattern_16 = {col_16}')
with open('pk_dir/cls_16.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_16,f)

# for col_17 in pattern_idx_17 :
# print(f'pattern_17 = {col_17}')
with open('pk_dir/cls_17.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_17,f)

# for col_18 in pattern_idx_18 :
# print(f'pattern_18 = {col_18}')
with open('pk_dir/cls_18.pickle', 'wb' ) as f :
    pickle.dump(pattern_idx_18,f)

# for col_19 in pattern_idx_19 :
# print(f'pattern_19 = {col_19}')
# with open('pk_dir/cls_19.pickle', 'wb' ) as f :
#     pickle.dump(pattern_idx_19,f)

# # for col_20 in pattern_idx_20 :
# # print(f'pattern_20 = {col_20}')
# with open('pk_dir/cls_20.pickle', 'wb' ) as f :
#     pickle.dump(pattern_idx_20,f)

# # for col_21 in pattern_idx_21 :
# # print(f'pattern_21 = {col_21}')
# with open('pk_dir/cls_21.pickle', 'wb' ) as f :
#     pickle.dump(pattern_idx_21,f)

# # for col_22 in pattern_idx_22 :
# # print(f'pattern_22 = {col_22}')
# with open('pk_dir/cls_22.pickle', 'wb' ) as f :
#     pickle.dump(pattern_idx_22,f)

# for i in range(22) :
#     name = 'pattern_idx_' + str(i+1)
#     for k in name :
#         print(f'list = {k}')
        # with open('pk_dir/cls_' + str(i+1) + '.pickle', 'wb') as f :
        #     pickle.dump(name, f) 
# y = np.zeros(len(total))

# X_train, X_temp, y_train, y_temp = train_test_split(total, y, test_size = 0.3, random_state = 42)
# X_valid, X_test, _, _ = train_test_split(X_temp, y_temp, test_size = 0.66, random_state = 42)  

# with open(os.path.join('CP_train.pickle'), 'wb') as f:
#    pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(os.path.join('CP_valid.pickle'), 'wb') as f:
#    pickle.dump(X_valid, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(os.path.join('CP_test.pickle'), 'wb') as f:
#    pickle.dump(X_test, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done.")