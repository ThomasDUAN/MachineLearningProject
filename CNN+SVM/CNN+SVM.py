# Due to some modifications of the Python interface in the latest Caffe version, 
# in order to run the code in this file please replace
#
# if ms != self.inputs[in_][1:]:
#   raise ValueError('Mean shape incompatible with input shape.')
#
# in caffe_root/python/caffe/io.py line 253-254, with
#
# if ms != self.inputs[in_][1:]:
#    print(self.inputs[in_])
#    in_shape = self.inputs[in_][1:]
#    m_min, m_max = mean.min(), mean.max()
#    normal_mean = (mean - m_min) / (m_max - m_min)
#    mean = resize_image(normal_mean.transpose((1,2,0)),
#              in_shape[1:]).transpose((2,0,1)) * \
#              (m_max - m_min) + m_min
#
# Reference: http://stackoverflow.com/questions/28692209/using-gpu-despite-setting-cpu-only-yielding-unexpected-keyword-argument


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/home/mshduan/Programs/caffe-master/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

files_path = './'
def get_mean_image(path):
    proto_obj = caffe.io.caffe_pb2.BlobProto()
    proto_file = open(path,'rb')
    proto_data = proto_file.read()
    proto_obj.ParseFromString(proto_data)
    means = np.asarray(proto_obj.data)
    return means.reshape(3,256,256)

# For Places-CNN
MODEL_FILE = files_path + 'places205CNN_deploy.prototxt'
PRETRAINED = files_path + 'places205CNN_iter_300000.caffemodel'
mean = get_mean_image(files_path + 'places205CNN_mean.binaryproto')

# For Hybrid-CNN
#MODEL_FILE = files_path + 'hybridCNN_deploy.prototxt'
#PRETRAINED = files_path + 'hybridCNN_iter_700000.caffemodel'
#mean = get_mean_image(files_path + 'hybridCNN_mean.binaryproto')

net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=mean, channel_swap = (2, 1, 0),raw_scale = 255, image_dims=(227, 227))
caffe.set_mode_cpu()

# Environment setup finished

########################

# This function converts the fc7 feature to the LIBSVM format
def dataToString(label, data):
	# To normalize feature vectors, uncomment the next line
    #datavec = data/np.linalg.norm(data)
    datavec = data
    datastr = label + ' ' 
    for i in range(len(datavec)):
        datastr += str(i+1) + ':' + str(datavec[i]) + ' '
    return datastr

import os 
# Create feature file for roadworks images (code can be reused for crosswalks images)       
fout1 = open('roadworks','w')
imagePath = "/home/mshduan/Desktop/Baustelle/"
label = '+1'
for image in sorted(os.listdir(imagePath)):
    input_image = caffe.io.load_image(imagePath+image)
    prediction = net.predict([input_image],oversample=False)
    fout1.write(dataToString(label,net.blobs['fc7'].data[0]) + '\n')
fout1.close()

# Create feature file for non-roadworks images  
fout2 = open('non_roadworks','w')
imagePath = "/home/mshduan/Desktop/ohne_Baustelle/"
label = '-1'
for image in sorted(os.listdir(imagePath)):
    input_image = caffe.io.load_image(imagePath+image)
    prediction = net.predict([input_image],oversample=False)
    fout2.write(dataToString(label,net.blobs['fc7'].data[0]) + '\n')   
fout2.close()    


########################
# normalize.py
########################
# normalize.py normalizes the feature vectors in a feature file
# usage: python normalize.py feature_file
# output: feature_file.normalized
!python normalize.py train_feature 

###########################################################
# Bash command, use LIVSVM to train and test svm classifier
###########################################################
!shuf train_feature -o train_data
!svm-train -c 32 -g 0.002 train_data  # returns the svm classifier "train_data.model"
!svm-predict test_1000 train_data.model test_1000.pre


########################
# accu.py 
########################
# accu.py calculates the accuracy for both roadworks(or crosswalks) and non-roadworks(or non-crosswalks) images
# usage: python accu.py test_target_file test_prediction_file 

!python accu.py test_1000 test_1000.pre

########################
# find.py
########################
# find.py finds the wrongly classified images
# usage: python find.py test_target_file test_prediction_file roadworks_feature_file non_roadworks_feature_file,
#        feature_files should not be shuffled, that means features in feature_files are sorted in alphabetical order according to the image names,
#        so that the original images can be found based on the indexs in the feature_files.   
# output: four lists, first two lists save indexes of correctly and incorrectly classified roadworks(or crosswalks) images repectively,
#                     next two lists save indexes of correctly and incorrectly classified non-roadworks(or non-crosswalks) images repectively.

!python find.py test_1000 test_1000.pre roadworks non_roadworks
# another example:
!python find.py test_1000 test_1000.pre crosswalks non_crosswalks

# copy the wrongly classified images 
import shutil
import os
srcpath = '/home/mshduan/Desktop/Baustelle/'
dst = '/home/mshduan/Desktop/Baustelle_wrong/'
count = 1
# l1 should be initialized with the list which saves indexes of incorrectly classified roadworks images, you can just copy the output of find.py
for image in sorted(os.listdir(srcpath)):
    if count in l1:
        shutil.copy(srcpath+image, dst)
        #print image
    count += 1	
    
 
