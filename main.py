import sys, argparse, os
import torch

from utils.log import Logger
from cityscapes.data_conf import labels

def parse_args():
    parser = argparse.ArgumentParser(description="A semantic segmentation program for cityscapes dataset")

    ###
    ### Program mode : [train, test]
    parser.add_argument("--mode", type=str, default="train",
            help="Choose the program mode : [train, test]")
    parser.add_argument("--model", type=str, default="deeplab",
            help="Select a network model among [deeplab, ...]")

    ###
    ### Training dataset config
    parser.add_argument("--num_class", type=int, default=len(labels)-1,
            help="Number of classes")
    parser.add_argument("--data_path", type=str, default="/home/jhson/mystudy/segmentation/dataset/CITYSCAPES",
            help="path to training dataset <Cityscapes>")
    parser.add_argument("--save_path", type=str, default="/data/segment/city/2/",
            help="path to save results (Weights, Images and logs)")
    parser.add_argument("--img_size", type=list, nargs='+', default=[300,580],
            help="Resize images for training to [img_size]")



    ###
    ### Test dataset config
    parser.add_argument("--model_path", type=str, default="/data/segment/city/3/model.pth",
            help="path to a trained model for inference")
    parser.add_argument("--test_path", type=str, default="/data/segment/city/test/",
            help="path to test images")
    parser.add_argument("--result_path", type=str, default="/data/segment/city/test/result/",
            help="path to save results of test images")


    ###
    ### Training policy
    parser.add_argument("--num_epoch", type=int, default=128, 
            help="The number of epoch")
    parser.add_argument("--batch_size", type=int, default=7, 
            help="The size of mini-batch")
    parser.add_argument("--lr", type=float, default=5e-4, 
            help="The learning rate")
    parser.add_argument("--pretrained", type=bool, default=True, 
            help="Load pretrained weights or not")
    parser.add_argument("--frequency_log", type=int, default=10, 
            help="The frequency for logging results")
    parser.add_argument("--frequency_image", type=int, default=100, 
            help="The frequency for saving images")


    ###
    ### Machine environment
    parser.add_argument("--num_cpu", type=int, default=5, 
            help="The number of cpu cores for the dataloader")
    parser.add_argument("--num_gpu", type=int, default=1, 
            help="The number of gpus for learning")



    return parser.parse_args()

def set_env(args):

    ### Check cuda-availability
    if torch.cuda.is_available() == False:
        args.num_gpu = 0

    ### Select a training device
    if args.num_gpu > 0:
        args.device = "cuda"
    else:
        args.device = "cpu"

    ### Make a folder for saving results   
    if os.path.isdir(args.save_path) == False:
        os.mkdir(args.save_path)


if __name__ == "__main__":    
    
    ###
    ### Parsing command line argumnets
    args = parse_args()

    ###
    ### Set program environment
    set_env(args)
    logger = Logger(args.save_path)
    logger.log(args)

    ###
    ### Train mode
    if args.mode.lower() == "train":
        
        ### deeplab model
        if args.model.lower() == "deeplab":
            from model.deeplab import Deeplab
            dlab = Deeplab(args, logger)
            dlab.train()

        else:
            print("[[[ERROR]]]] <model> should be [deeplab]")
        
    ### Test mode
    elif args.mode.lower() == "test":
        pass
    ### Mode selection error
    else:
        print("[[[ERROR]]]] <mode> should be [train, test]")


