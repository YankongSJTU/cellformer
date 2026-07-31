#import openslide
import argparse
import random
import os
import cv2
import numpy as np
import sys
import glob
import shutil
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
#from skimage.morphology import watershed
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def histeq2(im,nbr_bins):
        im2=np.float32(im-im.min())*np.float32(nbr_bins)/np.float32(im.max()-im.min())
        return im2

def color_region(grayimg):
    img1=grayimg
    a=img1
    image = a
    distance = ndi.distance_transform_edt(image)
    
    _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)
    #local_maxi = peak_local_max(distance, indices=False, min_distance=35,  labels=image)
    distance=cv2.GaussianBlur(255-m,(3,3),1)
    local_maxi = peak_local_max(distance, indices=False, min_distance=19, footprint=np.ones((25,25)), labels=image)
    markers = ndi.label(local_maxi)[0]
    #### merge markers
    labels = watershed(-distance, markers, mask=image)
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    circleimg=img1.copy()
    k=1
    nums=0
    img5=np.zeros(img1.shape[0]*img1.shape[1]*3)
    img5=img5.reshape(img1.shape[0],img1.shape[1],3)
    for i in range(2+maxval):
        tmplabel=labels.copy()
        if i >1 :
            tmplabel=np.maximum(0,(i-tmplabel))
            tmplabel2=np.maximum(0,(2-tmplabel))
            tmplabel=np.multiply(tmplabel,tmplabel2)
            tmplabel=tmplabel*(i-1)
            tmplabel=histeq2(tmplabel,255)
            contours,hier=cv2.findContours(np.uint8(tmplabel), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for num in range(len(contours)):
                r = random.randint(0, 255)
                g= random.randint(0, 255)
                b = random.randint(100, 255)
                c=[]
                c.append( contours[num])
                img5=cv2.fillPoly(img5, pts =c, color=(r, g, b))
    return(img5)

def array_region(grayimg):
    img1=grayimg
    a=img1
    image = a
    image=image
    distance = ndi.distance_transform_edt(image)
    _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)
    #local_maxi = peak_local_max(distance, indices=False, min_distance=35,  labels=image)
    distance=cv2.GaussianBlur(255-m,(3,3),1)
    local_maxi = peak_local_max(distance, indices=False, min_distance=19, footprint=np.ones((25,25)), labels=image)
    markers = ndi.label(local_maxi)[0]
    #### merge markers
    labels = watershed(-distance, markers, mask=image)
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    circleimg=img1.copy()
    k=1
    nums=0
    img5=np.zeros(img1.shape[0]*img1.shape[1])
    img5=img5.reshape(img1.shape[0],img1.shape[1])
    fillarray=[]
    for i in range(2+maxval):
        tmplabel=labels.copy()
        if i >1 :
            tmplabel=np.maximum(0,(i-tmplabel))
            tmplabel2=np.maximum(0,(2-tmplabel))
            tmplabel=np.multiply(tmplabel,tmplabel2)
            tmplabel=tmplabel*(i-1)
            tmplabel=histeq2(tmplabel,255)
            contours,hier=cv2.findContours(np.uint8(tmplabel), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for num in range(len(contours)):
                c=[]
                c.append( contours[num])
                tmpfig=img5.copy()
                tmpfig=cv2.fillPoly(tmpfig, pts =c, color=1)
                fillarray.append(tmpfig)
                
    _, thresh = cv2.threshold(np.uint8(distance),1,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # overlapped part   
   # _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)
    #local_maxi = peak_local_max(distance, indices=False, min_distance=35,  labels=image)
    distance=cv2.GaussianBlur(255-m,(3,3),1)
    local_maxi = peak_local_max(distance, indices=False, min_distance=10, footprint=np.ones((25,25)), labels=image)
    markers = ndi.label(local_maxi)[0]
    #### merge markers
    labels = watershed(-distance, markers, mask=image)
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    circleimg=img1.copy()
    k=1
    nums=0
    img5=np.zeros(img1.shape[0]*img1.shape[1])
    img5=img5.reshape(img1.shape[0],img1.shape[1])
    fillarray2=[]
    for i in range(2+maxval):
        tmplabel=labels.copy()
        if i >1 :
            tmplabel=np.maximum(0,(i-tmplabel))
            tmplabel2=np.maximum(0,(2-tmplabel))
            tmplabel=np.multiply(tmplabel,tmplabel2)
            tmplabel=tmplabel*(i-1)
            tmplabel=histeq2(tmplabel,255)
            contours,hier=cv2.findContours(np.uint8(tmplabel), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for num in range(len(contours)):
                c=[]
                c.append( contours[num])
                tmpfig=img5.copy()
                tmpfig=cv2.fillPoly(tmpfig, pts =c, color=1)
                fillarray2.append(tmpfig)
    for i in range(len(fillarray)):
        for j in range(len(fillarray2)):
            if(np.sum(np.multiply(fillarray[i],fillarray2[j]))>50):
                fillarray[i]=fillarray[i]+fillarray2[j]
                print("ok")
    return(fillarray)
    # return an list with many two-dimensional array, array number is the predict contour number

def cal_jaccard_index(gray1,gray2):
    #ret, image1 = cv2.threshold(gray1,0,1,cv2.THRESH_BINARY)
    #ret, image2 = cv2.threshold(gray2,0,1,cv2.THRESH_BINARY)
    m=np.multiply(gray1,gray2)
    img3=gray1+gray2
    _,img3=cv2.threshold(img3,0,1,cv2.THRESH_BINARY)
    n=np.sum(m)
    ji=np.float32(np.sum(m)/n)
    return(ji)
    
    
    
def cal_aji_1(each_gt_object):
    val=0
    index=-1
    result1=0
    result2=0
    for i in range(len(pred_array)):
        if(np.sum(np.multiply(pred_array[i],each_gt_object))>0):
            tmp_jaccard_index=cal_jaccard_index(pred_array[i],each_gt_object)
            if(tmp_jaccard_index>val):
                val=tmp_jaccard_index
                index=i
    if index >=0:
        result1=np.sum(np.multiply(each_gt_object,pred_array[index]))
        print(len(pred_array),index)
        _,tmpval=cv2.threshold((each_gt_object+pred_array[index]),0,1,cv2.THRESH_BINARY)
        result2=np.sum(tmpval)
        del pred_array[index]
    else:
        _,tmpval=cv2.threshold(each_gt_object,0,1,cv2.THRESH_BINARY)
        result2=np.sum(tmpval)
        result1=0
    return(result1,result2)


def cal_aji_2(gt_anno):
    all_pf=0
    for i in range(len(pred_array)):
        if(np.sum(np.multiply(pred_array[i],gt_anno))==0):
            all_pf=all_pf+np.sum(pred_array[i])
    return(all_pf)

def cal_aji(gt_anno,gt_path):
    file_glob=os.path.join(gt_path+'*.'+'png')
    file_list = []
    file_list.extend(glob.glob(file_glob))
    aj_upper=0
    aj_down_1=0
    aj_down_2=0
    newarray=pred_array
    
    #print(aj_down_2)
    for file in file_list:
        file.strip()
        img=cv2.imread(file,0)
        ret, image = cv2.threshold(img,0,1,cv2.THRESH_BINARY)
        ## eacho gt region pixel is 1 in 'image'
        if(np.sum(image)<5000):
            tmp1,tmp2=cal_aji_1(image)
            print(tmp1,tmp2)
            aj_upper=aj_upper+tmp1
            aj_down_1=aj_down_1+tmp2
            #print(tmp1,tmp2)
       #print(tmp2)
    aj_down_2=cal_aji_2(gt_anno)      
      
    aji=aj_upper/(aj_down_1+aj_down_2)   
    return(aji)
    

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def channel_change(img,w,h):
    tmp=np.zeros(w*h*3)
    tmp=tmp.reshape([h,w,3])
    tmp[:,:,0]=img[:,:,2]
    tmp[:,:,2]=img[:,:,0]
    tmp[:,:,1]=img[:,:,1]
    return(tmp)

def slit_small_level_0(line2,annoimg,arr,piece_num,piece_sizeW,piece_sizeH):
    filename=str.split(os.path.basename(os.path.splitext(line2)[0]),'_level1')[0]
    pathname=os.path.dirname(line2)
    slidename=pathname+"/"+filename+".svs"
#    print(slidename)
    #print(arr)
    if len(arr)>10:
        piece_num=10
    if len(arr)>20:
        piece_num=20
    if len(arr)>30:
        piece_num=30
#    if len(arr)>40:
#        piece_num=40
    source=openslide.OpenSlide(slidename)
    [w,h]=source.level_dimensions[0]
    [w1,h1]=source.level_dimensions[1]
    if w1>4000:
        if len(arr)>piece_num:
            if os.path.exists('./pred/small_pieces/'+filename):
                shutil.rmtree('./pred/small_pieces/'+filename)
            os.makedirs("./pred/small_pieces/"+filename)
            selarr=random.sample(arr,piece_num)
            for (i,j) in selarr:
                y=int(i*w/w1)
                x=int(j*h/h1)
                region=np.array(source.read_region((x,y),0,(piece_sizeW,piece_sizeH)))
                tmp=channel_change(region,piece_sizeW,piece_sizeH)
                #tmp=region
        #        tmpimg=annoimg[i*1000:(i+1)*1000,j*1000:(j+1)*1000]
        #        raw=cv2.imread('./inputs/TCGACOAD/images/'+filename2+".png",-1)
                cv2.imwrite('./pred/small_pieces/'+filename+"/"+filename+"_"+str(i)+"_"+str(j)+".png",tmp)
    return()

def conformback(gray1,gray2):
    ## merge gray2 into gray1
    tmp=gray1.copy()
    _,tmp1=cv2.threshold(gray1,250,1,cv2.THRESH_BINARY)    
    _,tmp2=cv2.threshold(gray2,250,1,cv2.THRESH_BINARY)    
    tmpover=np.multiply(tmp1,tmp2)
    uniq1=np.multiply(tmp1,(1-tmp2))
    # gray1 uniq
    uniq2=np.multiply(tmp2,(1-tmp1))
    # gray2 uniq
    gray3=gray1/2+gray2/2
    grayover=np.multiply(gray3,tmpover)
    
    grayresult=grayover+np.multiply(uniq1,gray1)+np.multiply(uniq2,gray2)
    return(grayresult)



def conform(img1,img2):
    ## img1 is ready to update area and may have some values
    _,tmp1=cv2.threshold(img1,0,1,cv2.THRESH_BINARY)
    _,tmp2=cv2.threshold(img2,0,1,cv2.THRESH_BINARY)
    overregion=np.multiply(tmp1,tmp2)
    uniq1=tmp1-overregion
    uniq2=tmp2-overregion
    tmp4=(np.multiply(img2,overregion)/2+np.multiply(img1,overregion)/2)
    tmp5=np.multiply(uniq1,img1)
    tmp6=np.multiply(uniq2,img2)
    return(tmp6+tmp4+tmp5)
def fillimage(img,h,w):
    newimg=np.zeros([h,w,3])
    h1=img.shape[0]
    w1=img.shape[1]
    newimg[0:h1,0:w1]=img
    return(newimg)
def cropimg(img,h,w):
    newimg=img[0:h,0:w]
    return(newimg)
def find_obj_contour2(grayimg,rawimg):
    img0=grayimg.copy()
    _,grayimg = cv2.threshold(np.uint8(grayimg),250,1,cv2.THRESH_BINARY)
    mindistance=11
    kval=3
    npones=11
    distance = ndi.distance_transform_edt(np.uint8(grayimg))
    _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)
    distance=cv2.GaussianBlur(255-m,(kval,kval),1)
    local_maxi = peak_local_max(distance,indices=False, min_distance=mindistance, footprint=np.ones((npones,npones)), labels=np.uint8(grayimg))
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=np.uint8(grayimg))
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    k=1
    nums=0
#    positions=[]
    img6=np.zeros([grayimg.shape[0],grayimg.shape[1],3])
    img6[:,:,0]=img0
    img6[:,:,1]=img0
    img6[:,:,2]=img0
    allarea=[]
    img5=img6.copy()
    #for j in range(val):
    allradius=[]
    allcontours=[]
    for j in range(maxval):
        i=j+1
        tmplabel=labels.copy()
        tmplabel[tmplabel!=i]=0
        tmplabel=histeq2(tmplabel,255)
        contours,hier=cv2.findContours(np.uint8(tmplabel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            (x,y),radius=cv2.minEnclosingCircle(contours[0])
#            positions.append([x,y])
            if(cv2.contourArea(contours[0])>10):
                allcontours.append( contours[0])
                nums=nums+1
                allarea.append(cv2.contourArea(contours[0]))
    updatedcontours=[]
    for j in range(len(allcontours)):
   #         if allarea[j]>(1/2)*np.median(allarea) and allarea[j]<2*np.median(allarea):
            maxrec=cv2.minAreaRect(allcontours[j])
            updatedcontours.append(allcontours[j])
            box=cv2.boxPoints(maxrec)
            box=np.int0(box)
            img6=cv2.drawContours(img6,[box],0,(0,0,255),2)
            r = random.randint(0, 255)
            g= random.randint(0, 255)
            b = random.randint(100, 255)
            c=[]
            c.append( allcontours[j])
            img5=cv2.fillPoly(img5, pts =c, color=(r, g, b))
    return(img5,nums,img6,updatedcontours)

def find_obj_contour3(grayimg,rawimg):
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


def find_obj_contour(grayimg,rawimg):
    img0=grayimg.copy()
    _,grayimg = cv2.threshold(np.uint8(grayimg),250,1,cv2.THRESH_BINARY)
    mindistance=13
    kval=3
    npones=13
    distance = ndi.distance_transform_edt(np.uint8(grayimg))
    _, thresh = cv2.threshold(np.uint8(distance),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    distance=np.multiply(distance,1-thresh)
    m=255-histeq2(distance,255)

#    local_maxi = peak_local_max(distance,indices=False, min_distance=mindistance, footprint=np.ones((npones,npones)), labels=np.uint8(grayimg))
    distance=cv2.GaussianBlur(255-m,(kval,kval),1)
    local_maxi = peak_local_max(distance, min_distance=mindistance, footprint=np.ones((npones,npones), dtype=np.bool_), labels=np.uint8(grayimg))
    mask=np.zeros(distance.shape,dtype=bool)
    mask[tuple(local_maxi.T)]=True
    markers,_=ndi.label(mask)
    labels = watershed(-distance, markers, mask=np.uint8(grayimg))

#    local_maxi = peak_local_max( distance, min_distance=mindistance, footprint=np.ones((npones, npones)), labels=np.uint8(grayimg))
 #   markers = ndi.label(local_maxi)[0]
  #  labels = watershed(-distance, markers, mask=np.uint8(grayimg))
    m=np.uint8(histeq2(labels,255))
    maxval=np.max(labels)
    k=1
    nums=0
    positions=[]
    img6=np.zeros([grayimg.shape[0],grayimg.shape[1],3])
    img6[:,:,0]=img0
    img6[:,:,1]=img0
    img6[:,:,2]=img0
    allarea=[]
    img5=img6.copy()
    allradius=[]
    allcontours=[]
    updatedradius=[]
    updatedpositions=[]
    for j in range(maxval):
        i=j+1
        tmplabel=labels.copy()
        tmplabel[tmplabel!=i]=0
        tmplabel=histeq2(tmplabel,255)
        contours,hier=cv2.findContours(np.uint8(tmplabel),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            (x,y),radius=cv2.minEnclosingCircle(contours[0])
            positions.append([x,y])
            allradius.append(radius)
            if(cv2.contourArea(contours[0])>10):
                allcontours.append( contours[0])
                allradius.append(radius)
                allarea.append(cv2.contourArea(contours[0]))
    updatedcontours=[]
    for j in range(len(allcontours)):
            if allarea[j]>(1/5)*np.median(allarea) and allarea[j]<5*np.median(allarea):
                maxrec=cv2.minAreaRect(allcontours[j])
                updatedcontours.append(allcontours[j])
                updatedradius.append(allradius[j])
                nums=nums+1
                box=cv2.boxPoints(maxrec)
                box=np.int0(box)
                img6=cv2.drawContours(img6,[box],0,(0,0,255),2)
                r = random.randint(0, 255)
                g= random.randint(0, 255)
                b = random.randint(100, 255)
                c=[]
                c.append( allcontours[j])
                img5=cv2.fillPoly(img5, pts =c, color=(r, g, b))
                updatedpositions.append(positions[j])
    return(img5,nums,img6,updatedcontours)#,updatedpositions,updatedradius)
    #return(img5,nums,img6,updatedcontours,updatedpositions,updatedradius)

