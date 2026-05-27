import torch
import numpy as np
from PIL import Image,ImageFilter
from torchvision import transforms
import torchvision.models as models
from torch.utils.data.dataset import Dataset 
import random 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = img.astype('float16') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float16') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, opt, data, size=56, max_cells=2500):
        self.X_nucpatch = data['x_nucpatch']  
#        self.X_samplenames = data['x_samplename']  
        self.X_samplenames = data['x_imgname']  
        self.X_labels = data['x_tumor']        
        self.X_nucpos = data['x_nucpatch_pos']        
        tumor_labels = list(set(self.X_labels))
        self.label_map = {
        "dataBLCA":1,
        "dataBRCA1":2,
        "BRCA1":2,
        "BRCA2":2,
        "BRCA":2,
        "dataCESC":3,
        "dataCHOL":4,
        "dataCOAD":5,
        "dataDLBC":6,
        "dataESCA":7,
        "dataGBM":8,
        "dataHNSC":9,
        "dataKICH":10,
        "dataKIRC":11,
        "SHENAI":11,
        "dataKIRP":12,
        "dataLGG":13,
        "dataLIHC":14,
        "dataLUAD":15,
        "dataLUSC":16,
        "dataOV":17,
        "dataPAAD":18,
        "dataPRAD":19,
        "dataREAD":20,
        "dataSTAD":21,
        "dataTHCA":22,
        "dataTHYM":23,
        "dataUCEC1":0,
        "UCEC1":0,
        "UCEC":0,
        "HER2":2
        }

        self.labels = [self.label_map[x] for x in self.X_labels]

        self.size = size
        self.max_cells = max_cells  
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        images_nps = self.X_nucpatch[index]  
        x_samplename = self.X_samplenames[index]
        x_nucpatchpos=self.X_nucpos[index]
        num_cells = images_nps.shape[0]
        label = self.labels[index]  

        if num_cells == 0:
            x_nucpatch = torch.zeros((0, 3, self.size, self.size))  
            return x_nucpatch, x_samplename, num_cells, label,torch.from_numpy(x_nucpatchpos)

        if num_cells > self.max_cells:
            selected_cells = np.random.choice(num_cells, self.max_cells, replace=False)
            images_nps = images_nps[selected_cells]
            x_nucpatchpos = x_nucpatchpos[selected_cells] 

        cell_patches = []
        for cell_idx, cell_patch in enumerate(images_nps):
            if cell_patch.dtype != np.uint8 and cell_patch.dtype != np.float32:
                cell_patch = cell_patch.astype(np.float32)
            if cell_patch.dtype == np.uint8:
                image_pil = Image.fromarray(cell_patch)
            else:
                cell_patch = (cell_patch - cell_patch.min()) / (cell_patch.max() - cell_patch.min()) * 255
                image_pil = Image.fromarray(cell_patch.astype(np.uint8))
            tensor = self.transform(image_pil)  
            cell_patches.append(tensor)

        x_nucpatch = torch.stack(cell_patches)  
        return x_nucpatch, x_samplename, len(cell_patches), label,torch.from_numpy(x_nucpatchpos)

    def __len__(self):
        return len(self.X_samplenames)
class aDatasetLoaderV2(torch.utils.data.Dataset):
    def __init__(self, data, size=56, max_cells=2500, is_train=True):
        self.patches = data['x_nucpatch']
        self.coords = data['x_nucpatch_pos']
        self.names = data['x_imgname']
        self.labels = data['x_tumor'] # 需根据 label_map 转换
        self.is_train = is_train
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180) if is_train else transforms.Lambda(lambda x: x),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_patches = self.patches[index] # (N, 56, 56, 3)
        img_coords = self.coords[index]
        
        # 采样逻辑
        num_cells = len(img_patches)
        if num_cells > 2500:
            idx = np.random.choice(num_cells, 2500, replace=False)
            img_patches = img_patches[idx]
            img_coords = img_coords[idx]
            
        # 批量应用转换
        tensors = torch.stack([self.transform(p) for p in img_patches])
        
        return tensors, self.names[index], len(img_patches), self.labels[index], torch.from_numpy(img_coords)
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DatasetLoaderV2(Dataset):
    def __init__(self, data, size=56, max_cells=2500, is_train=True):
        """
        data: 加载自 CreateDataset.py 生成的 pkl 文件
        size: 细胞 Patch 的缩放尺寸 (默认 56)
        max_cells: 每张图最多处理的细胞数量 (防止 OOM)
        """
        self.X_nucpatch = data['x_nucpatch']      # [N, 56, 56, 3] 的列表
        self.X_nucpos = data['x_nucpatch_pos']    # [N, 2] 的列表
        self.X_imgname = data['x_imgname']
        self.X_labels = data['x_tumor']           # 肿瘤类型字符串
        
        self.size = size
        self.max_cells = max_cells
        self.is_train = is_train

        # 肿瘤类别映射 (请根据你的实际情况修改)
        self.label_map = {
            "UCEC": 0, "UCEC1":0,"BLCA": 1, "BRCA": 2, "CESC": 3, "CHOL": 4, 
            "COAD": 5, "DLBC": 6, "ESCA": 7, "GBM": 8, "HNSC": 9,
            "KICH": 10, "KIRC": 11, "KIRP": 12, "LGG": 13, "LIHC": 14,
            "LUAD": 15, "LUSC": 16, "OV": 17, "PAAD": 18, "PRAD": 19,
            "READ": 20, "STAD": 21, "THCA": 22, "THYM": 23
        }

        # 图像增强：加入旋转以增强 CPSformer 的旋转不变性特性
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180) if is_train else transforms.Lambda(lambda x: x),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # 1. 获取当前图的所有细胞 Patch 和坐标
        patches = self.X_nucpatch[index]  # np.uint8, [N, 56, 56, 3]
        poses = self.X_nucpos[index]      # [N, 2]
        img_name = self.X_imgname[index]
        label_str = self.X_labels[index]
        label = self.label_map.get(label_str, 0)

        num_cells = patches.shape[0]

        # 2. 细胞下采样 (防止单张图细胞过多导致显存溢出)
        if num_cells > self.max_cells:
            selected_idx = np.random.choice(num_cells, self.max_cells, replace=False)
            patches = patches[selected_idx]
            poses = poses[selected_idx]
            num_cells = self.max_cells

        # 3. 对每个细胞 Patch 进行预处理
        # 这种批量 transform 是最耗时的，但为了数据增强必须做
        cell_tensors = torch.stack([self.transform(p) for p in patches])

        # 4. 返回数据
        # 返回：细胞张量, 掩码(在 collate 中处理), 图像名, 类别标签, 原始坐标
        return cell_tensors, img_name, num_cells, label, torch.from_numpy(poses)

    def __len__(self):

        return len(self.X_imgname)
class DatasetLoaderV3(Dataset):
    def __init__(self, data, size=56, max_cells=2500, is_train=True):
        """
        data: 加载自 CreateDataset.py 生成的 pkl 文件
        size: 细胞 Patch 的缩放尺寸 (默认 56)
        max_cells: 每张图最多处理的细胞数量 (防止 OOM)
        """
        self.X_nucpatch = data['x_nucpatch']      # [N, 56, 56, 3] 的列表
        self.X_nucpos = data['x_nucpatch_pos']    # [N, 2] 的列表
        self.X_imgname = data['x_imgname']
        self.X_labels = data['x_tumor']           # 肿瘤类型字符串
        
        self.size = size
        self.max_cells = max_cells
        self.is_train = is_train

        # 肿瘤类别映射 (请根据你的实际情况修改)
        self.label_map = {
            "UCEC": 0, "BLCA": 1,"BRCA1":2,"BRCA2":2, "BRCA": 2, "CESC": 3, "CHOL": 4, 
            "COAD": 5, "DLBC": 6, "ESCA": 7, "GBM": 8, "HNSC": 9,
            "KICH": 10, "KIRC": 11, "KIRP": 12, "LGG": 13, "LIHC": 14,
            "LUAD": 15, "LUSC": 16, "OV": 17, "PAAD": 18, "PRAD": 19,
            "READ": 20, "STAD": 21, "THCA": 22, "THYM": 23
        }

        # 图像增强：加入旋转以增强 CPSformer 的旋转不变性特性
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180) if is_train else transforms.Lambda(lambda x: x),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # 1. 获取当前图的所有细胞 Patch 和坐标
        patches = self.X_nucpatch[index]  # np.uint8, [N, 56, 56, 3]
        poses = self.X_nucpos[index]      # [N, 2]
        img_name = self.X_imgname[index]
        label_str = self.X_labels[index]
        label = self.label_map.get(label_str, 0)

        num_cells = patches.shape[0]

        # 2. 细胞下采样 (防止单张图细胞过多导致显存溢出)
        if num_cells > self.max_cells:
            selected_idx = np.random.choice(num_cells, self.max_cells, replace=False)
            patches = patches[selected_idx]
            poses = poses[selected_idx]
            num_cells = self.max_cells

        # 3. 对每个细胞 Patch 进行预处理
        # 这种批量 transform 是最耗时的，但为了数据增强必须做
        cell_tensors = torch.stack([self.transform(p) for p in patches])

        # 4. 返回数据
        # 返回：细胞张量, 掩码(在 collate 中处理), 图像名, 类别标签, 原始坐标
        return cell_tensors, img_name, num_cells, label, torch.from_numpy(poses)

    def __len__(self):

        return len(self.X_imgname)
