import pickle
from glob import glob

# pickle_list = sorted(glob('pk_dir/cls*.pickle'))
pickle_list = sorted(glob('pk_dir/cls_*.pickle'))
sum_data = 0
for pk_name in pickle_list :
    with open(pk_name, "rb") as f:
        data = pickle.load(f)
        print(f'pk_name = {pk_name}, cls_count = {len(data)}')
        sum_data +=len(data)

# print(f'total = {sum_data}')

# with open('pk_dir/CP_test.pickle', "rb") as f :
#     data = pickle.load(f)
# count = 0
# for contents in data :
#     if contents[5] == "10" :
#         count+=1
# print(count)
# with open('pk_dir/cls_2.pickle', 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
    # print(len(data))
# with open('CP_train.pickle', 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
# with open('CP_valid.pickle', 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))


# print("!!")
# print(data)
# print(len(data))