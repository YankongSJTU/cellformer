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
        self.X_samplenames = data['x_samplename']  
        self.X_labels = data['x_tumor']        
        self.X_nucpos = data['x_nucpatch_pos']        
        tumor_labels = list(set(self.X_labels))
        self.label_map = {
        "dataBLCA":1,
        "dataBRCA1":2,
        "BRCA1":2,
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
        "dataUCEC1":0
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
