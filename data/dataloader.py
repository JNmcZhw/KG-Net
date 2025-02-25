import glob
import itertools
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as tf
import os

random.seed(41)

################################################# 训练数据加载器 ##################################################

class TrainDataloader(Dataset):

    def __init__(self, haze_path, clear_path, transform=None,unaligned=False, model="train"):

        self.transform = tf.Compose(transform)
        self.unaligned = unaligned
        self.model = model

        self.haze_path = os.path.join(haze_path,"*.*")
        self.clear_path = os.path.join(clear_path, "*.*")

        self.list_haze = sorted(glob.glob(self.haze_path))
        self.list_clear = sorted(glob.glob(self.clear_path))

        print("Total {} examples:".format(model), max(len(self.list_haze), len(self.list_clear)))


    def __getitem__(self, index):

        if self.unaligned:

            haze = self.list_haze[random.randint(0, len(self.list_haze) - 1)]

        else:

            haze = self.list_haze[index % len(self.list_haze)]

        clear = self.list_clear[index % len(self.list_clear)]

        name = os.path.basename(clear)

        haze = Image.open(haze).convert("RGB")
        clear = Image.open(clear).convert("RGB")

        haze = self.transform(haze)
        clear = self.transform(clear)

        return haze,clear,name

    def __len__(self):

        return max(len(self.list_haze),len(self.list_clear))



class TestDataloader(Dataset):

    def __init__(self, haze_path, clear_path, transform=None, model="test"):

        self.transform = tf.Compose(transform)
        self.model = model

        self.haze_path = os.path.join(haze_path,"*.*")
        self.clear_path = os.path.join(clear_path,"*.*")

        self.list_haze = sorted(glob.glob(self.haze_path))
        self.list_clear = sorted(glob.glob(self.clear_path))

        print("Total {} examples:".format(model), max(len(self.list_haze), len(self.list_clear)))


    def __getitem__(self, index):

        haze = self.list_haze[index % len(self.list_haze)]

        clear = self.list_clear[index % len(self.list_clear)]

        name = os.path.basename(haze)

        haze = Image.open(haze).convert("RGB")
        clear = Image.open(clear).convert("RGB")

        # w = haze.size[0]
        # h = haze.size[1]
        # size = [h,w]

        haze = self.transform(haze)
        clear = self.transform(clear)

        # return haze, clear, name,size
        return haze, clear, name

    def __len__(self):

        return max(len(self.list_haze),len(self.list_clear))




if  __name__ == "__main__":
     haze_path= "../results/stylized150/"
     clear_path= "../results/stylized150/"
     style_path= "../results/"
     transform_ = [tf.ToTensor()]

     train_sets=TrainDataloader(haze_path,clear_path,style_path,transform_)

     dataload = DataLoader(train_sets,batch_size=1,shuffle=True,num_workers=4)

     for i, batch in enumerate(dataload):

         print((batch[8].shape))