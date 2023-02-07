from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GenderImgDataset(Dataset):
    ''' Dataset for gender classification. '''
    def __init__(self, root_dir, phase):
        self.root_dir = root_dir
        self.phase = phase
        self.file_path = root_dir + phase #./genderImg/test
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((225,225)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.575, 0.550, 0.535], [0.241, 0.238, 0.240])
            ]),
            'val': transforms.Compose([
                transforms.Resize((225,225)),
                transforms.ToTensor(),
                transforms.Normalize([0.575, 0.550, 0.535], [0.241, 0.238, 0.240])
            ]),
            'test' : transforms.Compose([
                transforms.Resize((225,225)),
                transforms.ToTensor(),
                transforms.Normalize([0.575, 0.550, 0.535], [0.241, 0.238, 0.240])
            ]),
        }
        self.image_path = glob.glob(self.file_path + '/*/*.jpg')

    def __len__(self):
        return len(self.image_path) #./genderImg/train/male/*.jpg

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert('RGB')
        image = self.transform[self.phase](image)

        if self.image_path[idx].split('/')[-2] == 'male':
            label = 0
        else :
            label = 1 
        if self.phase == 'test':
            return image, label, self.image_path[idx]
        else:
            return image, label


if __name__ == "__main__":
    testset = GenderImgDataset('./genderImg/', 'test')
    print(len(testset.image_path))
    
