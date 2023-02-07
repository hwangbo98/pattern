from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

class GenderImgDataset(Dataset):
    ''' Dataset for gender classification. '''
    def __init__(self, root_dir, phase):
        self.root_dir = root_dir
        self.phase = phase
        # self.file_path = root_dir + phase #./genderImg/test
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((225,225)),
                # transforms.Resize((225,225)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.RandomGrayscale(p=0.3),
                transforms.Normalize([0.535, 0.490, 0.470], [0.229, 0.225, 0.222])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((225,225)),
                transforms.ToTensor(),
                transforms.Normalize([0.535, 0.490, 0.470], [0.229, 0.225, 0.222])
            ]),
            'test' : transforms.Compose([
                transforms.Resize((225,225)),
                transforms.ToTensor(),
                transforms.Normalize([0.535, 0.490, 0.470], [0.229, 0.225, 0.222])
            ]),
        }
        # self.tag = self.root_dir.split('/')[3]
        # if self.tag == "showniq_croped":
        #     self.pickle = 'showniq_' + self.phase + '.pickle'   #cropped images
        # elif self.tag == "face_blur_dataset":
        #     self.pickle = 'showniq_blur_' + self.phase + '.pickle'    #cropped and blurred images for train
        # elif self.tag == "blurpadding":
        #     self.pickle = 'showniq_blurpadding_' + self.phase + '.pickle'    #cropped, blurred, and padded images for train
        # elif self.tag == "padding":
        #     self.pickle = 'showniq_padding_' + self.phase + '.pickle'    #cropped and padded images
        self.pickle = root_dir + '/CP_' + phase +".pickle"
        with open(self.pickle, 'rb') as f:
            data = pickle.load(f)
        # def genderfilter(x):
        #    return x.split('/')[-4] == 'men'
        # data = list(filter(genderfilter, data))
        
        # self.image_path = [path[0] for path in data]
        self.total_data = data

    def __len__(self):
        return len(self.total_data) #./genderImg/train/male/*.jpg

    def __getitem__(self, idx):
        image = Image.open(self.total_data[idx][0]).convert('RGB')
        area = (self.total_data[idx][1],self.total_data[idx][2],self.total_data[idx][3],self.total_data[idx][4])
        image = image.crop((area))
        image = self.transform[self.phase](image)

        label = int(self.total_data[idx][5]) - 1
             
        if self.phase == 'test':
            return image, label, self.total_data[idx][0]
        else:
            return image, label


if __name__ == "__main__":
    testset = GenderImgDataset('/home/hwangbo/color_pattern/pk_dir', 'train')
    print(testset.__getitem__(4))
    # images = {}
    # for img in testset.image_path:
    #     if img.split('/')[-3] not in images:
    #         images[img.split('/')[-3]] = 1
    #     else:
    #         value = images[img.split('/')[-3]]
    #         del images[img.split('/')[-3]]
    #         images[img.split('/')[-3]] = value + 1
    # print(images)



