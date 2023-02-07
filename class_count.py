import pickle


with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
    # print(data)
    print(len(data))

dict_cls = {}

for arr in data :
    if arr[6] in dict_cls :
        dict_cls[arr[6]] +=1
    else :
        dict_cls[arr[6]] = 1

sorted_keys = sorted(dict_cls.items(), key=lambda x : int(x[0]))
sorted_values = sorted(dict_cls.items(), key=lambda x : int(x[1]), reverse=True)
print(f'sum ={sum(dict_cls.values())}')
print(f'dict_cls = {sorted_values} & cls_num = {len(dict_cls.items())}')

