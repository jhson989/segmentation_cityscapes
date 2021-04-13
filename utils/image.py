import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from cityscapes.data_conf import labels

def convert_rgb(pred):
    
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for label in labels:
        id = label.id
        color = label.color
        pred_rgb[pred==id] = color

    pred_rgb = Image.fromarray(pred_rgb)
    return pred_rgb



def save_image(name, real, gt, pred):
    real, gt, pred = real[0].cpu(), gt[0].cpu(), pred[0].cpu()

    real = transforms.ToPILImage()(real)
    gt = np.uint8(gt.detach().numpy())
    pred = np.uint8(torch.argmax(pred, dim=0).detach().numpy())

    gt_rgb = convert_rgb(gt)
    pred_rgb = convert_rgb(pred)


    imgs = [real, gt_rgb, pred_rgb]

    widths, heights = zip(*(i.size for i in imgs))
    totalWidth = sum(widths)
    totalHeight = max(heights)

    new_img = Image.new("RGB", (totalWidth, totalHeight))
    offset = 0
    for img in imgs:
        new_img.paste(img, (offset, 0))
        offset += img.size[0]

    new_img.save(name)
        



