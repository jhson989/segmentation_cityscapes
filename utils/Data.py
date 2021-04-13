import torch
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image


class ToTensor(object):
    def __call__(self, target):
        target = np.array(target)
        '''
        new_target = np.zeros_like(target)
        for idx in [24]:
            new_target[target==idx] = 1
        for idx in [26,27,28,29,30,31]:
            new_target[target==idx] = 2
        for idx in [7]:
            new_target[target==idx] = 3
        target = torch.as_tensor(new_target, dtype=torch.int64)
        '''
        target[target==-1] = 0
        target = torch.as_tensor(target, dtype=torch.int64)

        return target



def get_dataloader(data_path="/home/jhson/mystudy/segmentation/dataset/CITYSCAPES", split="train", batch_size=1, num_cpu=1, img_size=(300, 580)):

    trf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    trf_mask = transforms.Compose([
        transforms.Resize(img_size),
        ToTensor(),
    ])
    dataset = Cityscapes(data_path, split=split, mode='fine',target_type='semantic', transform=trf, target_transform=trf_mask)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu-1)

    return data_loader

if __name__ == "__main__":

    data_loader = get_dataloader()

    for idx, (img, smnt) in enumerate(data_loader):
        print( "%d / %d : %s, %s "% (idx, len(data_loader), str(img.shape), str(smnt.shape) ) )
        img = transforms.ToPILImage()(img[0])
        img.show()
        break



