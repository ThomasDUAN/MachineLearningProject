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

MODEL_FILE = files_path + 'finetuned_places205CNN_deploy.prototxt'
PRETRAINED = files_path + 'places205CNN_det_finetune_iter_3000.caffemodel'
mean = get_mean_image(files_path + 'places205CNN_mean.binaryproto')

net = caffe.Classifier(MODEL_FILE, PRETRAINED,mean=mean, channel_swap = (2, 1, 0),raw_scale = 255, image_dims=(256, 256))
caffe.set_mode_cpu()

import os
# test the fine-tuned model on roadworks images
imagePath = "/home/mshduan/Programs/caffe-master/examples/construction_cnn/test/bau/"
pred_list=[]
for image in sorted(os.listdir(imagePath)):
    input_image = caffe.io.load_image(imagePath+image)
    prediction = net.predict([input_image],oversample=True)
    pred_list.append(prediction[0].argmax())

# number of false positives predictions    
len(filter(lambda x: x==0,pred_list))

# test the fine-tuned model on non-roadworks images
imagePath = "/home/mshduan/Programs/caffe-master/examples/construction_cnn/test/nothing/"
pred_list_2=[]
for image in sorted(os.listdir(imagePath)):
    input_image = caffe.io.load_image(imagePath+image)
    prediction = net.predict([input_image],oversample=True)
    pred_list_2.append(prediction[0].argmax())

###########################################
# copy the incorrectly classified images    
###########################################
import shutil
import os
srcpath = '/home/mshduan/Programs/caffe-master/examples/construction_cnn/test/bau/'
dst = '/home/mshduan/Desktop/cons_cnn/wrong_bau/'
count = 0
for image in sorted(os.listdir(srcpath)):
    if pred_list[count] == 0:
        shutil.copy(srcpath+image, dst)
    count += 1	    

srcpath = '/home/mshduan/Programs/caffe-master/examples/construction_cnn/test/nothing/'
dst = '/home/mshduan/Desktop/cons_cnn/wrong_noting/'
count = 0
for image in sorted(os.listdir(srcpath)):
    if pred_list_2[count] == 1:
        shutil.copy(srcpath+image, dst)
    count += 1	
