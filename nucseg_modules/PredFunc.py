from __future__ import print_function
import scipy.misc as misc
import cv2

import os
import matplotlib.pyplot as plt
from six.moves import xrange
import datetime

import numpy as np
import tensorflow as tf
import time
#import denseCRF
import TensorflowUtils as Utils
from matplotlib.colors import ListedColormap, BoundaryNorm

# Hide the warning messages about CPU/GPU
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mode_test(sess,batch_size, net, test_dataset_reader, pred_annotation, score_final_train,image, annotation, training, train_op, loss, summary_op, saver):
#def mode_test(sess,batch_size, net, test_dataset_reader, pred_annotation, score_final_train,image, annotation, training, train_op, loss, summary_op, summary_writer, saver):
    print(">>>>>>>>>>>>>>>> real images")
    start = time.time()
    steps=test_dataset_reader.get_num_of_records()
    for i in xrange(steps):
        test_images,_,filename=test_dataset_reader.next_batch_test(batch_size)
        feed_dict={image:test_images,training:False}
        
        pred2= sess.run( pred_annotation, feed_dict=feed_dict)
        pred2=np.squeeze(pred2)

        j=0
        pred=pred2
       # filebasename=os.path.basename(filename[j])
        filebasename=str.split(os.path.basename(filename[j]),'.')[0]
        cv2.imwrite("./pred/"+filebasename+'.png',pred)
    end = time.time()
#
#
    print("Learning time:", end - start, "seconds")
#
#
