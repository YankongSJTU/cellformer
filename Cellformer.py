import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from models import *
from utils.DataSets import *
from utils.utils import *
import os
import pickle
import argparse
from torch.utils.data.dataset import Dataset 
from torchvision import transforms
import torch.nn.parallel
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdatadir', default='data200', help="datasets")
    parser.add_argument('--gpu_ids', type=str, default='6', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--batch_size', type=int, default=12, help="Number of batches to train/test for. Default: 1")
    parser.add_argument('--patchsize', type=int, default=56, help="Resize cell patch images into * pixels. Default: 26")
    opt = parser.parse_known_args()[0]
    return opt
opt = parse_args()
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(opt.gpu_ids[0]))
os.environ['CUDA_VISIBLE_DEVICES'] =str(opt.gpu_ids)

if __name__ == "__main__":
    opt = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=MILCellModelmerge()
    model = model.cuda()
    if os.path.isfile(opt.checkpoints_dir+"/model.pth"):
        checkpoint=torch.load(opt.checkpoints_dir+"/model.pth",map_location='cuda')
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.') and not isinstance(model, nn.DataParallel):
                new_state_dict[k[7:]] = v  # 移除 'module.'
            elif not k.startswith('module.') and isinstance(model, nn.DataParallel):
                new_state_dict['module.' + k] = v  # 添加 'module.'
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
                module.eval()
        data_test_path =opt.testdatadir
        testdata = pickle.load(open(data_test_path, 'rb'))
        custom_data_loader_test = DatasetLoader(opt, testdata,size=opt.patchsize)
        test_loader = torch.utils.data.DataLoader( dataset=custom_data_loader_test, batch_size=opt.batch_size, shuffle=True,drop_last=False,num_workers=3,collate_fn=custom_collate_fn)
        basename="task1.model1.testdata.small0206"+".cell_feature"
        namelist,featurelist=evaluate(opt,test_loader,model,device)
        totalname=namelist
        totalfeature=featurelist
        df = pd.DataFrame(totalfeature, index=totalname)
        csv_filename = opt.testdatadir+'_group_feature0206.csv'
        df.to_csv(csv_filename, index_label='samplename')
    else:
        print("No saved model!")
        exit()
