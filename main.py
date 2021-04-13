import sys, argparse, os
import torch
import torch.nn as nn
import torch.optim as optim

from model.Deeplab import get_custom_DeepLabv3
from utils.Data import get_dataloader 
from utils.image import save_image
from utils.labels import labels

def parse_args():
    parser = argparse.ArgumentParser(description="Deeplabv3 training")

    parser.add_argument("--mode", type=str, default="train", help="")

    parser.add_argument("--num_class", type=int, default=len(labels)-1, help="")
    parser.add_argument("--weight_class", type=list, nargs='+', default=[1,2,2,2], help="")
    parser.add_argument("--data_path", type=str, default="/home/jhson/mystudy/segmentation/dataset/CITYSCAPES", help="")
    parser.add_argument("--save_path", type=int, default=2, help="")

    parser.add_argument("--num_epoch", type=int, default=128, help="")
    parser.add_argument("--batch_size", type=int, default=7, help="")
    parser.add_argument("--lr", type=float, default=5e-4, help="")

    parser.add_argument("--num_cpu", type=int, default=5, help="")
    parser.add_argument("--num_gpu", type=int, default=1, help="")

    return parser.parse_args()

def set_env(args):

    if torch.cuda.is_available() == False:
        args.num_gpu = 0
    if args.num_gpu > 0:
        args.device = "cuda"
    else:
        args.device = "cpu"

    args.save_path = "/data/segment/city/"+str(args.save_path)+"/"
    if os.path.isdir(args.save_path) == False:
        os.mkdir(args.save_path)


if __name__ == "__main__":    
    
    args = parse_args()
    set_env(args)

    dlab = get_custom_DeepLabv3(args.num_class).to(args.device)
    dataloader = get_dataloader(args.data_path, args.mode, args.batch_size, args.num_cpu)
#    crit_CE = nn.CrossEntropyLoss(weight=torch.Tensor(args.weight_class)).to(args.device)
    crit_CE = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(dlab.parameters(), lr=args.lr)

    for epoch in range(args.num_epoch):
        for idx, (img, smnt) in enumerate(dataloader):
            img, smnt = img.to(args.device), smnt.to(args.device)
            
            optimizer.zero_grad()
            pred = dlab(img)["out"]
            loss = crit_CE(pred, smnt)
            loss.backward()
            optimizer.step()
            print("[ [%4d/%4d] [%4d/%4d] ] Training CE(%.3f)" % (epoch, args.num_epoch, idx, len(dataloader),loss.item()))

            if (idx%100==0) :
                img = save_image(args.save_path, "%d-%d.jpg"%(epoch,idx), img, smnt, pred, args.num_class)





