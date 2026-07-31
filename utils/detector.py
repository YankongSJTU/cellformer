import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

class NucleiDetector:
    def __init__(self, patch_size=56, radius=28):
        self.patch_size = patch_size
        self.radius = radius

    def detect_and_crop(self, image_rgb):
        """Detect and crop cell nuclei patches from RGB image."""
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Distance transform and watershed to locate centers
        distance = ndi.distance_transform_edt(thresh)
        local_maxi = peak_local_max(distance, min_distance=10, labels=thresh, footprint=np.ones((7, 7)))
        
        patches = []
        coords = []
        h, w, _ = image_rgb.shape
        
        for pos in local_maxi:
            y, x = pos
            # Boundary-safe cropping
            y1, y2 = max(0, int(y-self.radius)), min(h, int(y+self.radius))
            x1, x2 = max(0, int(x-self.radius)), min(w, int(x+self.radius))
            
            crop = image_rgb[y1:y2, x1:x2]
            # Pad edges that are too small
            if crop.shape[0] < self.radius*2 or crop.shape[1] < self.radius*2:
                crop = cv2.copyMakeBorder(crop, 0, self.radius*2-crop.shape[0], 
                                          0, self.radius*2-crop.shape[1], cv2.BORDER_REFLECT)
            
            patch_resized = cv2.resize(crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            patches.append(patch_resized)
            coords.append([x, y])
            
        return np.array(patches), np.array(coords)
