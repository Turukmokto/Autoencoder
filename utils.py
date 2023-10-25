import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import os
import random

B_COEF = [2, 4]
Test_B_COEF = [2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EvalDataset(Dataset):
    def __init__(self, filenames, transform=None):
        self.all_filenames = filenames
        self.sz = len(self.all_filenames)
        self.transform = transform

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.all_filenames[idx]
        image = Image.open(img_name).convert('RGB')
        res = self.transform(image)
        image.close()
        return res


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def make_dataloader(path, BATCH_SIZE):
    list_imgs_names = get_list_images(path)
    list_imgs_names_train, list_imgs_names_test = list_imgs_names[:int(len(list_imgs_names) * 0.9)], \
        list_imgs_names[int(len(list_imgs_names) * 0.9):]
    image_datasets = {'Train': list_imgs_names_train, 'Test': list_imgs_names_test}
    dataset = {'Train': EvalDataset(list_imgs_names_train, transform),
               'Test': EvalDataset(list_imgs_names_test, transform)}
    dataloader = {'Train': DataLoader(dataset['Train'], shuffle=True, batch_size=BATCH_SIZE),
                  'Test': DataLoader(dataset['Test'], shuffle=True, batch_size=BATCH_SIZE), }
    return image_datasets, dataloader


def get_list_images(path):
    list_imgs_names = []
    for label_name in os.listdir(f'{path}'):
        for file_name in os.listdir(f'{path}/{label_name}'):
            list_imgs_names.append(f'{path}/{label_name}/{file_name}')
    random.shuffle(list_imgs_names)
    return list_imgs_names
