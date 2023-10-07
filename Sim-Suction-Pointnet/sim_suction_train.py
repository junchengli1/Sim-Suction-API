import argparse
import os
import pickle
import sys
current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_directory)
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import open3d
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time
from sim_suction_dataset import SimSuctionDataset

import utils
from torch.utils.data import DataLoader
from sim_suction_model.sim_suction_pointnet import ScoreNet

from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
base_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description='SimSuctionPointNet')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', type=str, default='training')
parser.add_argument('--batch-size', type=int, default=2)#16)#16
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr-score' , type=float, default=0.001) #0.001
parser.add_argument('--lr-region', type=float, default=0.001)

parser.add_argument('--model_path', type=str, default= base_dir / 'models', help='to saved model path')
parser.add_argument('--log-path', type=str, default=base_dir / 'log/', help='to saved log path')
parser.add_argument('--data_root', type=str, default="../pointcloud_train", help='point cloud path')
parser.add_argument('--suction_label_root', type=str, default="../sim_suction_label", help='suction label path')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--save-interval', type=int, default=1)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False
data_root=args.data_root
suction_label_root=args.suction_label_root

def replace_in_file(file_path, original_str, replacement_str):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    # Ensure the replacement doesn't change the file's length (important for binary files)
    assert len(original_str) == len(replacement_str), "Replacement strings must be of equal length"
    
    # Replace content
    new_data = file_data.replace(original_str.encode(), replacement_str.encode())
    
    with open(file_path, 'wb') as f:
        f.write(new_data)

np.random.seed(int(time.time()))
if args.cuda:
    torch.cuda.manual_seed(1)
torch.cuda.set_device(args.gpu)


logger = utils.mkdir_output(args.log_path, args.tag, args.mode, log_flag=True)


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

score_model = ScoreNet(training=True).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    score_model = nn.DataParallel(score_model)

score_model = score_model.to(device)



TRAIN_DATASET = SimSuctionDataset(data_root,suction_label_root)
    
print(len(TRAIN_DATASET))

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
    num_workers=2, drop_last=True, worker_init_fn=my_worker_init_fn,pin_memory=True)


resume_epoch=0
optimizer_score, scheduler_score = utils.construct_scheduler(score_model, args.lr_score, resume_epoch)



class SimSuctionPointnet():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 

        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def train_main(self, epoch, mode='train'):
        if mode == 'train':
            score_model.train()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader
            batch_size = args.batch_size

        total_all_loss = 0
        for batch_idx, (pc, suction_score) in enumerate(TRAIN_DATALOADER):
            if mode == 'train':
                optimizer_score.zero_grad()

            pc = pc.data.numpy()

            pc = torch.Tensor(pc)
            pc, suction_score = pc.float().cuda(), suction_score.long().cuda()
            _, _, loss = score_model(pc, suction_score)

            loss_total = loss.sum() 
            total_all_loss += loss.mean()
            if mode == 'train':
                loss_total.backward()
                optimizer_score.step()

            data = (loss,)
            utils.add_log_batch(logger, data, batch_idx + epoch * len(dataloader), mode=mode, method="score")
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(mode, epoch, 
                        batch_idx * batch_size, len(dataloader.dataset), 100. * batch_idx * 
                        batch_size / len(dataloader.dataset), loss_total.data, args.tag))
        data = (total_all_loss.data/batch_idx,)
        utils.add_log_epoch(logger, data, epoch, mode=mode, method="score")
        
        if mode == 'train':
            scheduler_score.step()

    
    
    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------training epoch", epoch, "------------------")
            path_score = os.path.join(args.model_path, 'score_{}.model'.format(epoch))
            self.train_main(epoch, mode='train')
            torch.save(score_model, path_score)

       



def main():
    if args.mode == 'training':
        scoreModule = SimSuctionPointnet(resume_epoch, args.epoch, TRAIN_DATALOADER)
        scoreModule.train()



if __name__ == "__main__":
    main()
