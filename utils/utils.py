import math
import random
import torch.nn.functional as F
import albumentations
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import MiniBatchKMeans
import joblib
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
import torch_geometric
import albumentations
from albumentations.augmentations import transforms
from tqdm import tqdm
from albumentations.core.composition import Compose
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset  
import cv2
import torch.utils.data
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score

class ImgDatasetmask(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir,segment_dir,num_classes=1, transform=None):
        self.img_ids = img_ids
        self.num_classes = num_classes
        self.transform = transform
        self.img_dir = img_dir
        self.segment_dir = segment_dir
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_id = os.path.basename(img_id).split('.')[0]
        img = cv2.imread(self.img_dir+ "/"+img_id+".png",-1 )
        img = img.astype(np.float32)
        imgmask = cv2.imread(self.segment_dir+ "/"+img_id+".png",0 )
        return imgmask,img 
def histeq2(im,nbr_bins):
        im2=np.float32(im-im.min())*np.float32(nbr_bins)/np.float32(im.max()-im.min())
        return im2

def get_patch_features(opt,imgfilelists):
    segment_dir="./data/"+opt.datadir+"/"+opt.nuc_seg_dir
    img_dir="./data/"+opt.datadir+"/"+opt.image_dir
    img_transform = Compose([ albumentations.Resize(1000,1000), transforms.Normalize(), ])
    img_dataset = ImgDatasetmask(img_ids=imgfilelists,img_dir = img_dir,segment_dir=segment_dir,transform=img_transform )
    img_loader = torch.utils.data.DataLoader(img_dataset,batch_size=100,shuffle=False,drop_last=False)
    objregions=[]
    allobjregions=[]
    allobjregions_pos=[]

    for input,  meta in tqdm(img_loader, total=len(img_loader)):
        for i in range(opt.piecenumber):

            objregions,poses=find_obj_contour(np.uint8(input.detach().numpy()[i]),np.uint8(meta.detach().numpy()[i]))
            for j in range(len(objregions)):
                allobjregions.append(objregions[j])
                allobjregions_pos.append(poses[j])
    return(allobjregions,allobjregions_pos)

def get_patch_featuresPred(opt, imgfilelists):
    segment_dir = "./data/" + opt.datadir + "/" + opt.nuc_seg_dir
    img_dir = "./data/" + opt.datadir + "/" + opt.image_dir
    batchsize = len(imgfilelists) if opt.allpatch == "all" else min(opt.piecenumber, len(imgfilelists))
    
    img_transform = Compose([
        albumentations.Resize(opt.patchsize, opt.patchsize),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    img_dataset = ImgDatasetmask(img_ids=imgfilelists, img_dir=img_dir, segment_dir=segment_dir, transform=img_transform)
    img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batchsize, shuffle=False, drop_last=False)
    
    objregions = []
    allobjregions = []
    allobjregions_pos = []
    allobjregions_no = []
    
    # Track processed image indices
    processed_img_indices = []
    
    for batch_idx, (input, meta) in enumerate(tqdm(img_loader, total=len(img_loader))):
        for i in range(batchsize):
            # Guard against last batch overflow
            if batch_idx * batchsize + i >= len(imgfilelists):
                continue
            
            objregions, poses = find_obj_contour(np.uint8(input.numpy()[i]), np.uint8(meta.numpy()[i]))
            if len(objregions) == 0:
                continue
                
            for j in range(len(objregions)):
                allobjregions.append(objregions[j])
                allobjregions_pos.append(poses[j])
            allobjregions_no.append(len(poses))
            
            # Record non-skipped image index
            processed_img_indices.append(batch_idx * batchsize + i)
    
    # Filter to processed images only
    updated_imgfilelists = [imgfilelists[idx] for idx in processed_img_indices]
    
    return allobjregions, allobjregions_pos, allobjregions_no, updated_imgfilelists
def get_patch_featuresPred2(opt,imgfilelists):
    segment_dir="./data/"+opt.datadir+"/"+opt.nuc_seg_dir
    img_dir="./data/"+opt.datadir+"/"+opt.image_dir
    batchsize = len(imgfilelists) if opt.allpatch == "all" else min(opt.piecenumber, len(imgfilelists))
    img_transform = Compose([
        albumentations.Resize(opt.patchsize, opt.patchsize),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img_dataset = ImgDatasetmask(img_ids=imgfilelists,img_dir = img_dir,segment_dir=segment_dir,transform=img_transform )
    img_loader = torch.utils.data.DataLoader(img_dataset,batch_size=batchsize,shuffle=False,drop_last=False)
    objregions=[]
    allobjregions=[]
    allobjregions_pos=[]
    allobjregions_no=[]
    for input,  meta in tqdm(img_loader, total=len(img_loader)):
        for i in range(batchsize):
            objregions,poses=find_obj_contour(np.uint8(input.numpy()[i]),np.uint8(meta.numpy()[i]))
            if len(objregions)==0:
                continue
            for j in range(len(objregions)):
                allobjregions.append(objregions[j])
                allobjregions_pos.append(poses[j])
            allobjregions_no.append(len(poses))
            continue
    return(allobjregions,allobjregions_pos,allobjregions_no)
def find_obj_contour(grayimg,rawimg):
    _,grayimg = cv2.threshold(np.uint8(grayimg),250,1,cv2.THRESH_BINARY)
    mindistance=11
    kval=3
    npones=11
    distance = ndi.distance_transform_edt(np.uint8(grayimg))
    _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)
    distance=cv2.GaussianBlur(255-m,(kval,kval),1)
    local_maxi = peak_local_max(distance, min_distance=mindistance, footprint=np.ones((npones,npones), dtype=np.bool_), labels=np.uint8(grayimg))
    mask=np.zeros(distance.shape,dtype=bool)
    mask[tuple(local_maxi.T)]=True
    markers,_=ndi.label(mask)
    labels = watershed(-distance, markers, mask=np.uint8(grayimg))
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    positions=[]
    nucleiregion=[]
    if maxval>3000:
        maxval=3000
    for j in range(maxval):
        i=j+1
        tmplabel=labels.copy()
        tmplabel[tmplabel!=i]=0
        tmplabel=histeq2(tmplabel,255)
        contours,hier=cv2.findContours(np.uint8(tmplabel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            (x,y),radius=cv2.minEnclosingCircle(contours[0])
            radius=40
            if radius>5:
                tmp=rawimg[max(0,int(y-radius)):min(1000,int(y+radius)),max(0,int(x-radius)):min(1000,int(x+radius))]
                tmp2=cv2.resize(tmp,(56,56),interpolation=(cv2.INTER_CUBIC))
                nucleiregion.append(tmp2)
                positions.append([x,y])
    return(nucleiregion,positions)

def random_select_nulceiregion(regions):
    selected_region=[]
    if(len(regions)>3000):
        selected_region=random.sample(regions,3000)
    else:
        selected_region=regions
        selected_region=selected_region+[random.choice(regions) for _ in range(3000-len(regions))]
    return(selected_region)

def getCellDataPred(opt, pat_name, pat2img):
    x_samplename,x_imgname, x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no = [],[],[],[],[]
    x_samplename=pat_name
    x_imgname=pat2img[pat_name]
    patchfeatures,patchfeatures_pos,patchfeatures_no=get_patch_featuresPred(opt,pat2img[pat_name])
    x_nuc_patches.append(np.array(patchfeatures))
    x_nuc_patches_pos.append(np.array(patchfeatures_pos))
    x_nuc_patches_no.append(np.array(patchfeatures_no))
    return x_samplename,x_imgname,x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no
def getCellData(opt, pat_name, pat2img):
    x_samplename,x_imgname, x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no = [],[],[],[],[]
    x_samplename=pat_name
    x_imgname=pat2img[pat_name]
    patchfeatures,patchfeatures_pos,patchfeatures_no,x_imgname=get_patch_featuresPred(opt,pat2img[pat_name])
    if not x_imgname:
        return [], [], [], [], []
    
    x_nuc_patches.append(np.array(patchfeatures))
    x_nuc_patches_pos.append(np.array(patchfeatures_pos))
    x_nuc_patches_no.append(np.array(patchfeatures_no))
    return x_samplename,x_imgname,x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no



def cal_loss5(predicted_features1, predicted_features2,ntxent_loss):

    loss_contrastive2 = cosine_similarity_loss(predicted_features1, predicted_features2)
    loss_contrastive = ntxent_loss(predicted_features1, predicted_features2)

    loss2 = torch.mean(torch.var(predicted_features1, dim=0))
    loss3 = torch.mean(torch.var(predicted_features2, dim=0))
    predicted_features1_norm = F.normalize(predicted_features1, dim=1)
    feature_mean = predicted_features1_norm.mean(dim=0, keepdim=True)
    covariance_matrix = torch.mm((predicted_features1_norm - feature_mean).T,
                                 (predicted_features1_norm - feature_mean)) / predicted_features1.size(0)
    off_diagonal = covariance_matrix - torch.diag(torch.diag(covariance_matrix))
    diversity_loss = torch.norm(off_diagonal, p='fro')
    diversity_loss = torch.clamp(diversity_loss, max=1.0)
    instance_loss = instance_diversity_loss(predicted_features1)
    return loss_contrastive,loss_contrastive2, loss2, loss3, diversity_loss,instance_loss

def instance_diversity_loss(features):
    normalized_features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
    batch_size = features.size(0)
    mask = torch.eye(batch_size, device=features.device).bool()
    diversity_loss = torch.mean(similarity_matrix[~mask] ** 2)
    return diversity_loss

def cosine_similarity_loss(x1, x2):
    x1_norm = F.normalize(x1, dim=1)
    x2_norm = F.normalize(x2, dim=1)
    cosine_sim = torch.sum(x1_norm * x2_norm, dim=1)
    loss = 1 - cosine_sim
    return loss.mean()

class NTXentLossback(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=torch.float32)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask.to(self.device)

    def forward(self, zi, zj):
        """
        zi and zj are the outputs from two different augmentations of the same batch.
        Each has shape [batch_size, embedding_dim]
        """
        N = 2 * self.batch_size
        zi = F.normalize(zi, dim=1)
        zj = F.normalize(zj, dim=1)
        representations = torch.cat([zi, zj], dim=0)  

        similarity_matrix = torch.matmul(representations, representations.T)  
        similarity_matrix = similarity_matrix / self.temperature
        labels = torch.arange(self.batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = self.mask

        positives = torch.cat([torch.diag(similarity_matrix, self.batch_size),
                               torch.diag(similarity_matrix, -self.batch_size)], dim=0).view(N, 1)

        negatives = similarity_matrix.masked_select(mask).view(N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(N).long().to(self.device)  

        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def mask_cell_features(cell_all_features, cell_all_poses, masks, image_width=1000, image_height=1000):
    batch_size = cell_all_poses.shape[0]
    masked_all_features = []
    masked_all_poses = []
    updated_masks = []
    max_new_num_cells = 0  
    for i in range(batch_size):
        cell_features = cell_all_features[i]  
        cell_poses = cell_all_poses[i]       
        mask = masks[i]                      
        num_cells = cell_features.size(0)
        if num_cells == 0 or mask.sum() == 0:  
            masked_features = torch.zeros(1, cell_features.size(1), cell_features.size(2), cell_features.size(3), device=cell_features.device)
            masked_poses = torch.zeros(1, cell_poses.size(1), device=cell_poses.device)
            updated_mask = torch.zeros(1, dtype=torch.bool, device=mask.device)
        else:
            retain_ratio = random.uniform(0.6, 0.99)
            region_size_x = int(image_width * retain_ratio)
            region_size_y = int(image_height * retain_ratio)
            center_x = random.randint(region_size_x // 2, image_width - region_size_x // 2)
            center_y = random.randint(region_size_y // 2, image_height - region_size_y // 2)
            rect_center = torch.tensor([center_x, center_y], device=cell_features.device)
            angle = random.uniform(0, 2 * math.pi)
            relative_positions = cell_poses - rect_center  
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            rotation_matrix = torch.tensor([[cos_angle, sin_angle], [-sin_angle, cos_angle]], device=cell_features.device, dtype=torch.float32)
            relative_positions = relative_positions.to(dtype=torch.float32)
            rotated_positions = torch.matmul(relative_positions, rotation_matrix.T)
            region_mask = (
                (rotated_positions[:, 0].abs() <= region_size_x / 2) &
                (rotated_positions[:, 1].abs() <= region_size_y / 2)
            )
            mask = mask.to(dtype=torch.bool)
            updated_mask = mask & region_mask
            if updated_mask.sum() == 0:  
                masked_features = torch.zeros(1, cell_features.size(1), cell_features.size(2), cell_features.size(3), device=cell_features.device)
                masked_poses = torch.zeros(1, cell_poses.size(1), device=cell_poses.device)
                updated_mask = torch.zeros(1, dtype=torch.bool, device=mask.device)
            else:
                masked_features = cell_features[updated_mask].clone()
                masked_poses = cell_poses[updated_mask].clone()
                masked_poses[:, 0] -= (center_x - region_size_x // 2)
                masked_poses[:, 1] -= (center_y - region_size_y // 2)
                masked_poses = torch.clamp(masked_poses, min=0, max=max(image_width, image_height))
                updated_mask = torch.ones(masked_features.size(0), dtype=torch.bool, device=mask.device)
        max_new_num_cells = max(max_new_num_cells, masked_features.size(0))
        masked_all_features.append(masked_features)
        masked_all_poses.append(masked_poses)
        updated_masks.append(updated_mask)
    for i in range(batch_size):
        if masked_all_features[i].size(0) < max_new_num_cells:
            padding = torch.zeros(max_new_num_cells - masked_all_features[i].size(0), *masked_all_features[i].shape[1:], device=masked_all_features[i].device)
            masked_all_features[i] = torch.cat([masked_all_features[i], padding], dim=0)
        if masked_all_poses[i].size(0) < max_new_num_cells:
            pose_padding = torch.zeros(max_new_num_cells - masked_all_poses[i].size(0), *masked_all_poses[i].shape[1:], device=masked_all_poses[i].device)
            masked_all_poses[i] = torch.cat([masked_all_poses[i], pose_padding], dim=0)

        if updated_masks[i].size(0) < max_new_num_cells:
            mask_padding = torch.zeros(max_new_num_cells - updated_masks[i].size(0), dtype=torch.bool, device=updated_masks[i].device)
            updated_masks[i] = torch.cat([updated_masks[i], mask_padding], dim=0)

    masked_all_features = torch.nn.utils.rnn.pad_sequence(masked_all_features, batch_first=True)
    masked_all_poses = torch.nn.utils.rnn.pad_sequence(masked_all_poses, batch_first=True)
    updated_masks = torch.nn.utils.rnn.pad_sequence(updated_masks, batch_first=True, padding_value=0)
    return masked_all_features, masked_all_poses, updated_masks


def cal_cls_auroc(pred, GT):
    predictions = np.array(pred)
    true_labels = np.array(GT)
    label_mapping = {
        1: "BLCA", 2: "BRCA1", 3: "CESC", 4: "CHOL", 5: "COAD", 6: "DLBC", 7: "ESCA",
        8: "GBM", 9: "HNSC", 10: "KICH", 11: "KIRC", 12: "KIRP", 13: "LGG", 14: "LIHC",
        15: "LUAD", 16: "LUSC", 17: "OV", 18: "PAAD", 19: "PRAD", 20: "READ", 21: "STAD",
        22: "THCA", 23: "THYM", 0: "UCEC1"
    }
    unique_labels = np.unique(true_labels)
    auroc_scores = {}
    for label in unique_labels:
        binary_true_labels = (true_labels == label).astype(int)
        binary_predictions = (predictions == label).astype(int)
        try:
            auroc = roc_auc_score(binary_true_labels, binary_predictions)
            auroc_scores[label] = {"auroc": auroc, "label_name": label_mapping[label]}
        except ValueError as e:
            print(f"Error calculating AUROC for label {label}: {e}")

    return auroc_scores


def test(opt, testloader, model, device):
    model.eval()
    allname = []
    allfeature = []
    total_correct = 0
    total_samples = 0
    alllabels = []
    allpred = []

    with torch.no_grad():
        for batch_idx, (inputs, masks, imgname, labels, pos) in enumerate(testloader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            pos=pos.to(device)
            if inputs.numel() == 0 or inputs.size(0) == 0:
                continue
            for i in range(len(labels)):
                alllabels.append(labels[i].item())
            group_features, cell_features, cls_logits = model(inputs, pos, masks)
            predicted_classes = torch.argmax(cls_logits, dim=1).cpu().detach().numpy()
            for i in range(len(predicted_classes)):
                allpred.append(predicted_classes[i].item())
            correct = (predicted_classes == labels.cpu()).sum().item()
            total = labels.size(0)
            total_samples += total
            total_correct += correct
            outputs = group_features.detach().cpu().numpy()
            masks = masks.cpu().numpy()
            for name, output in zip(imgname, outputs):
                if abs(output[0])>=0:
                    allname.append(name)
                    allfeature.append(output)
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy: {overall_accuracy:.4f}")
    return allname, allfeature, overall_accuracy, alllabels, allpred
def evaluate(opt, testloader, model, device):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
            module.eval()
    allname = []
    allfeature = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, masks, imgname, labels, pos) in enumerate(testloader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            pos=pos.to(device)
            if inputs.numel() == 0 or inputs.size(0) == 0:
                continue
            group_features, cell_features, cls_logits = model(inputs, pos, masks)
            outputs = group_features.detach().cpu().numpy()
            for name, output in zip(imgname, outputs):
                if abs(output[0])>=0:
                    allname.append(name)
                    allfeature.append(output)
    return allname, allfeature
import cv2
import numpy as np
import torch
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

class NucleiDetector:
    def __init__(self, patch_size=56, radius=28):
        self.patch_size = patch_size
        self.radius = radius

    def detect_and_crop(self, image_rgb):
        """Detect cell nuclei in RGB image and crop patches.

        Args:
            image_rgb: RGB image (H, W, 3)

        Returns:
            cell_patches: (N, 56, 56, 3) array
            coordinates: (N, 2) array of cell centroids
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        distance = ndi.distance_transform_edt(thresh)
        local_maxi = peak_local_max(distance, min_distance=10, labels=thresh, footprint=np.ones((7, 7)))
        
        patches = []
        coords = []
        h, w, _ = image_rgb.shape
        
        for pos in local_maxi:
            y, x = pos
            y1, y2 = max(0, int(y-self.radius)), min(h, int(y+self.radius))
            x1, x2 = max(0, int(x-self.radius)), min(w, int(x+self.radius))
            
            crop = image_rgb[y1:y2, x1:x2]
            if crop.shape[0] < self.radius*2 or crop.shape[1] < self.radius*2:
                crop = cv2.copyMakeBorder(crop, 0, self.radius*2-crop.shape[0], 
                                          0, self.radius*2-crop.shape[1], cv2.BORDER_REFLECT)
            
            patch_resized = cv2.resize(crop, (self.patch_size, self.patch_size))
            patches.append(patch_resized)
            coords.append([x, y])
            
        return np.array(patches), np.array(coords)

def get_cell_data_robust(opt, image_path):
    """Read image and detect cells directly (replaces getCellData)."""
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detector = NucleiDetector()
    patches, coords = detector.detect_and_crop(img)
    
    return {
        "patches": patches,  # (N, 56, 56, 3)
        "coords": coords,    # (N, 2)
        "count": len(coords)
    }
import torch

def custom_collate_fn(batch):
    """Collate function for DataLoader with variable-length cell data.

    Args:
        batch: list of tuples (cell_tensors, img_name, num_cells, label, poses)
    """
    # cell_tensors: [N, 3, 56, 56]
    # poses: [N, 2]
    cell_tensors_list, img_names, num_cells_list, labels, poses_list = zip(*batch)
    
    max_cells = max(num_cells_list)
    batch_size = len(cell_tensors_list)
    
    _, C, H, W = cell_tensors_list[0].shape
    
    padded_patches = torch.zeros((batch_size, max_cells, C, H, W))
    padded_poses = torch.zeros((batch_size, max_cells, 2))
    masks = torch.zeros((batch_size, max_cells), dtype=torch.bool)
    
    for i, (patches, poses, num_cells) in enumerate(zip(cell_tensors_list, poses_list, num_cells_list)):
        padded_patches[i, :num_cells] = patches
        
        padded_poses[i, :num_cells] = poses
        
        masks[i, :num_cells] = True
        
    batched_labels = torch.tensor(labels, dtype=torch.long)
    
    # [B, N_max, 3, 56, 56], [B, N_max], list, [B], [B, N_max, 2]
    return padded_patches, masks, list(img_names), batched_labels, padded_poses
def plot_loss_curve(log_path, save_dir, epoch):
    """Read CSV training log and generate loss curve plot."""
    try:
        df = pd.read_csv(log_path)
        df_batch = df[df['batch'] != 'AVG'].copy()
        df_batch['total_loss'] = pd.to_numeric(df_batch['total_loss'])

        plt.figure(figsize=(10, 6))
        plt.plot(df_batch['total_loss'], label='Total Loss', color='blue', alpha=0.5)
        if 'l_con' in df.columns:
            plt.plot(pd.to_numeric(df_batch['l_con']), label='Contrastive Loss', color='red', alpha=0.8)
        
        plt.title(f"Training Loss Curve up to Epoch {epoch}")
        plt.xlabel("Steps (Batch-wise)")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(save_dir, f"loss_curve_epoch_{epoch}.png"))
        plt.close()
        print(f"Loss curve updated: loss_curve_epoch_{epoch}.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    def forward(self, zip1, zip2):
            device = zip1.device
            N = zip1.shape[0]

            queries = torch.cat([zip1, zip2], dim=0)
            sim_matrix = torch.matmul(queries, queries.T) / self.temperature

            mask = torch.eye(2 * N, dtype=torch.bool).to(device)
            sim_matrix = sim_matrix.masked_fill(mask, -1e4)

            diag_mask = torch.eye(2 * N, dtype=torch.bool).to(device)
            pos_mask = torch.diag(torch.ones(N), N).to(device).bool()
            pos_mask_rev = torch.diag(torch.ones(N), -N).to(device).bool()

            pos_sim = torch.diag(sim_matrix, N)
            pos_sim_reverse = torch.diag(sim_matrix, -N)
            positives = torch.cat([pos_sim, pos_sim_reverse], dim=0).view(2 * N, 1)

            neg_mask = ~(diag_mask | pos_mask | pos_mask_rev)
            negatives = sim_matrix[neg_mask].view(2 * N, -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(2 * N).to(device).long()

            loss = self.criterion(logits, labels)
            return loss / (2 * N)


# ─────────────────────────────────────────────────────────────────────────────
#   Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
#
# ─────────────────────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Args:
        temperature: temperature scaling for similarity distribution (default 0.1).
        base_temperature: base temperature from the paper for gradient
            stabilization (default 0.07).

    forward(features, labels):
        features : [N, D] embeddings (L2 normalized internally).
        labels   : [N] class labels per sample.

    Multi-view mode: concatenate features/labels from multiple views along
    the batch dimension before calling forward.
    """
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        if features.dim() != 2:
            raise ValueError(f"features must be [N, D], got {features.shape}")
        N = features.size(0)

        features = F.normalize(features, dim=1)

        logits = torch.matmul(features, features.T) / self.temperature

        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        logits = logits.masked_fill(self_mask, -1e4)

        labels = labels.view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = label_mask * (~self_mask).float()

        pos_count = pos_mask.sum(1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return (logits.sum() * 0.0)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count.clamp(min=1.0)

        loss = -(self.base_temperature / self.temperature) * mean_log_prob_pos

        loss = loss[valid].mean()
        return loss
