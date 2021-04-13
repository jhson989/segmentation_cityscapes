
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from cityscapes.dataloader import get_dataloader_train 
from cityscapes.data_conf import labels
from utils.image import save_image

def get_custom_DeepLabv3(out_channels, pretrained):

    deeplab = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    deeplab.classifier[4] = nn.Conv2d(256, out_channels, (1,1), (1,1))

    deeplab.train()
    return deeplab


class Deeplab():

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
       
    def train(self):

        args = self.args
        logger = self.logger

        ###
        ### Network model
        self.model = get_custom_DeepLabv3(args.num_class, args.pretrained).to(args.device)
        self.model.train()
        ###
        ### Dataloader
        self.dataloader, self.dataloader_val = get_dataloader_train(args)
        
        ###
        ### Training Policy
        # criterion
        crit_ce = nn.CrossEntropyLoss().to(args.device)
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)


        #####
        ### Training iteration
        for epoch in range(args.num_epoch):

            #####
            ### Train
            avg_loss = 0.0
            self.model.train()
            for idx, (img, gt) in enumerate(self.dataloader):

                ### learning
                img, gt = img.to(args.device), gt.to(args.device)
                optimizer.zero_grad()
                pred = self.model(img)["out"]
                loss = crit_ce(pred, gt)
                loss.backward()
                optimizer.step()

                ### Logging
                avg_loss = avg_loss + loss.item()
                if idx % args.frequency_log == 0 and idx != 0: 
                    logger.log("[[%4d/%4d] [%4d/%4d]] loss CE(%.3f)" 
                            % (epoch, args.num_epoch, idx, len(self.dataloader), avg_loss/args.frequency_log))
                    avg_loss = 0.0
                if idx % args.frequency_image == 0: 
                    img_name = args.save_path + "/%d-%d.jpg"%(epoch,idx)
                    save_image(img_name, img, gt, pred)


            #####
            ### Validation
            avg_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for idx, (img, gt) in enumerate(self.dataloader_val):
                    img, gt = img.to(args.device), gt.to(args.device)
                    pred = self.model(img)["out"]
                    loss = crit_ce(pred, gt)
                    avg_loss = avg_loss + loss.item()

                ### Logging
                logger.log("[EVAL] [[%4d/%4d] loss CE(%.3f)" 
                        % (epoch, args.num_epoch, avg_loss/len(self.dataloader_val)))
                img_name = args.save_path + "/eval-%d.jpg"%(epoch)
                save_image(img_name, img, gt, pred)


            #####
            ### Save the trained model
            self.save_model(epoch)


    def save_model(self, epoch):
        pth_name = self.args.save_path + "/%d.pth"%epoch
        torch.save(self.model.state_dict(), pth_name)


