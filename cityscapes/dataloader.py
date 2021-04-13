import torch
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image


class ToTensor_for_mask(object):
    def __call__(self, target):
        target = np.array(target)
        target[target==-1] = 0
        target = torch.as_tensor(target, dtype=torch.int64)
        return target

def get_dataloader_train(args):

    ###
    ### Dataset config
    data_path = args.data_path
    split = args.mode
    batch_size = args.batch_size
    num_cpu = args.num_cpu
    img_size = args.img_size

    ###
    ### Image config & Augmentation policy
    trf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    trf_mask = transforms.Compose([
        transforms.Resize(img_size),
        ToTensor_for_mask(),
    ])


    ###
    ### Train Dataset load using torchvision
    dataset = Cityscapes(data_path, split=split, mode='fine',target_type='semantic', transform=trf, target_transform=trf_mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu, drop_last=True)


    ###
    ### Validation Dataset load using torchvision
    dataset_val = Cityscapes(data_path, split="val", mode='fine',target_type='semantic', transform=trf, target_transform=trf_mask)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_cpu, drop_last=True)



    return dataloader, dataloader_val



