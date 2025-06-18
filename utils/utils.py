import math
import umap
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

def get_patch_featuresPred(opt,imgfilelists):
    segment_dir="./data/"+opt.datadir+"/"+opt.nuc_seg_dir
    img_dir="./data/"+opt.datadir+"/"+opt.image_dir
    batchsize = len(imgfilelists) if opt.allpatch == "all" else min(opt.piecenumber, len(imgfilelists))
    img_transform = Compose([
        albumentations.Resize(opt.patchsize, opt.patchsize),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img_dataset = ImgDatasetmask(img_ids=imgfilelists,img_dir = img_dir,segment_dir=segment_dir,transform=img_transform )
    img_loader = torch.utils.data.DataLoader(img_dataset,batch_size=batchsize,shuffle=False,drop_last=True)
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
    patchfeatures,patchfeatures_pos,patchfeatures_no=get_patch_featuresPred(opt,pat2img[pat_name])
    x_nuc_patches.append(np.array(patchfeatures))
    x_nuc_patches_pos.append(np.array(patchfeatures_pos))
    x_nuc_patches_no.append(np.array(patchfeatures_no))
    return x_samplename,x_imgname,x_nuc_patches,x_nuc_patches_pos,x_nuc_patches_no

def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)

    if elem_type is torch_geometric.data.data.Data:
        return [Batch.from_data_list(samples, []) for samples in transposed]
    else:
        return [default_collate(samples) for samples in transposed]


def parse_gpuids(opt):
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
def cal_GAP(data):
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_clusters = 3
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    kmeans_actual = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_actual.fit(data)
    inertia_actual = kmeans_actual.inertia_
    
    n_random_datasets = 10
    random_inertias = []
    
    for _ in range(n_random_datasets):
        random_data = np.random.rand(n_samples, n_features)
        kmeans_random = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_random.fit(random_data)
        random_inertias.append(kmeans_random.inertia_)
        gap = np.mean(np.log(random_inertias) - np.log(inertia_actual))
    return(gap)
def cluster_plot(data,device,imagename,n_clsuters):
    neighbor=18
    batch_size = 100  
    max_iter = 100  
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter)
    mbkmeans.fit(data)
    cluster_assignments = mbkmeans.labels_
    cluster_centers = mbkmeans.cluster_centers_
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(data)
    cluster_labels = mbkmeans.labels_
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    num_classes = n_clusters
    cmap = ListedColormap(colors[:num_classes])
    plt.figure(figsize=(18, 16))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap=cmap, s=5)
    plt.colorbar()
    plt.title("UMAP Visualization of Mini-Batch K-Means Clusters")
    plt.show()
    plt.savefig('cluster.png',bbox_inches='tight')
    return(cluster_labels)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, zi, zj):
        """
        zi 和 zj 是来自同一批次不同增强视图的输出，形状为 [batch_size, embedding_dim]
        """
        batch_size = zi.size(0)
        N = 2 * batch_size
        zi = F.normalize(zi, dim=1)
        zj = F.normalize(zj, dim=1)
        representations = torch.cat([zi, zj], dim=0)  
        similarity_matrix = torch.matmul(representations, representations.T)  
        similarity_matrix = similarity_matrix / self.temperature
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.ones((N, N), dtype=torch.float32).to(self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        positives = torch.cat([torch.diag(similarity_matrix, batch_size),
                               torch.diag(similarity_matrix, -batch_size)], dim=0).view(N, 1)
        negatives = similarity_matrix.masked_select(mask.bool()).view(N, -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(N).long().to(self.device)  
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
def custom_collate_fn(batch):
    x_nucpatches, x_samplenames, num_cells_list, labels, poses = zip(*batch)
    max_cells = max(num_cells_list)
    padded_patches = []
    padded_poses = []  
    masks = []
    for x_nucpatch, pose, num_cells in zip(x_nucpatches, poses, num_cells_list):
        padding = torch.zeros((max_cells - num_cells, x_nucpatch.shape[1], x_nucpatch.shape[2], x_nucpatch.shape[3]))
        padded_patch = torch.cat([x_nucpatch, padding], dim=0)
        padded_patches.append(padded_patch)
        pose_padding = torch.zeros((max_cells - num_cells, 2))
        padded_pose = torch.cat([pose, pose_padding], dim=0)
        padded_poses.append(padded_pose)

        mask = torch.cat([torch.ones(num_cells,dtype=torch.bool), torch.zeros(max_cells - num_cells,dtype=torch.bool)])
        masks.append(mask)
    batched_patches = torch.stack(padded_patches)  
    batched_poses = torch.stack(padded_poses)  
    batched_masks = torch.stack(masks)  
    batched_samplenames = list(x_samplenames)
    batched_labels = torch.tensor(labels, dtype=torch.long)  

    return batched_patches, batched_masks, batched_samplenames, batched_labels, batched_poses


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

class NTXentLoss(nn.Module):
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
