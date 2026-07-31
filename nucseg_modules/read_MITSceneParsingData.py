 
 
# coding=utf-8
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
 
import TensorflowUtils as utils
 
# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
 
def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    #
    #     SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]  # ADEChallengeData2016
    result = create_image_lists(os.path.join(data_dir))
    print("Pickling ...")
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    # else:
    #     print("Found pickle file!")
 
    with open(pickle_filepath, 'rb') as f:  #
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result
 
    return training_records, validation_records
 
 
'''
  
  image_list{ 
           "training":[{'image': image_full_name, 'annotation': annotation_file, 'image_filename': },......],
           "validation":[{'image': image_full_name, 'annotation': annotation_file, 'filename': filename},......]
           }
'''
 
 
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}
 
    for directory in directories:  
        file_list = []
        image_list[directory] = []
 
        
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob)) 
 
        if not file_list:
            print('No files found')
        else:
            for f in file_list:  
                
                filename = os.path.splitext(f.split('/')[-1])[0]  
                annotation_file = str(image_dir)+"annotations/"+str(directory)+"/"+str(filename)+'.png'
                #print(annotation_file)

                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])  
        no_of_images = len(image_list[directory])  
        print('No. of %s files: %d' % (directory, no_of_images))
 
    return image_list
 
 

def read_datasetpred(data_dir):
    result = create_image_lists_pred(os.path.join(data_dir))
    records = result['pred']
    return records
#read_dataset(data_dir)
def aread_datasetpred(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)

    result = create_image_lists_pred(os.path.join(data_dir))
    print("Pickling ...")
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    #else:
     #    print("Found pickle file!")
 
    with open(pickle_filepath, 'rb') as f:  
        result = pickle.load(f)
        records = result['pred']
        del result
 
    return records
 
def create_image_lists_pred(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directory = ['pred']
    image_list = {}
    file_list = []
    image_list[directory[0]] = []
    file_glob = os.path.join(image_dir, '*.' + 'jpg')
    file_list.extend(glob.glob(file_glob))  
    if not file_list:
        print('No files found')
    else:
        for f in file_list:  #
            filename = os.path.splitext(f.split('/')[-1])[0]  # 
            #record = {'image': f, 'filename': filename}#  ima
            record = {'image': f, 'filename': filename,'annotation':f}#  ima
            image_list[directory[0]].append(record)
    no_of_images = len(image_list[directory[0]])  
    print('No. of %s files: %d' % (directory, no_of_images))
 
    return image_list
	
#read_dataset(data_dir)
