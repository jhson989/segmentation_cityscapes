import torch.nn as nn
from torchvision import models

def get_custom_DeepLabv3(out_channels):


    deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
    deeplab.classifier[4] = nn.Conv2d(256, out_channels, (1,1), (1,1))
    print(out_channels)

    deeplab.train()
    return deeplab


if __name__ == "__main__":
    deeplab = get_custom_DeepLabv3(12)
    print(deeplab)
    
