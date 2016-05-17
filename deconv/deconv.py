import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['image.cmap'] = 'gray'

# Make sure that the extended version of caffe is on the python path:
caffe_root = '/home/mshduan/Programs/caffe-master_4/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()

# normal CNN, initialized with fine-tuned Place-CNN after 3000 iterations
net = caffe.Net('deconv_finetuned_places205CNN_deploy.prototxt', 'places205CNN_det_finetune_iter_3000.caffemodel', caffe.TEST)
#net = caffe.Net('deconv_places205CNN_deploy.prototxt', '/home/zlinc/Programs/caffe-master/models/placesCNN/places205CNN_iter_300000.caffemodel', caffe.TEST)

# deconvnet
invnet = caffe.Net('invdeploy.prototxt',caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
def get_mean_image(path):
    proto_obj = caffe.io.caffe_pb2.BlobProto()
    proto_file = open(path,'rb')
    proto_data = proto_file.read()
    proto_obj.ParseFromString(proto_data)
    means = np.asarray(proto_obj.data)
    return means.reshape(3,256,256)
    
transformer.set_mean('data', get_mean_image('places205CNN_mean.binaryproto'))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


    
# This function visualizes the features of a image
def feat_vis(image):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    out = net.forward()

    for b in invnet.params:
        invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)
    
    feat = net.blobs['pool5'].data

    invnet.blobs['pooled'].data[...] = feat
    invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
    invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
    invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
    invnet.forward()

    feat = invnet.blobs['conv1'].data[0]
    feat -= feat.min()
    feat /= feat.max()
    plt.imshow(feat.transpose((1, 2,0)))
    image_name = image.split('/')[-1]
    # save the visualization, make sure that the following directory exists
    plt.savefig("cons_deconv_3000iter/"+image_name[:18]+"_deconv.png", dpi = 400, bbox_inches='tight', transparent=True)

# Make sure that all the selected images are in this directory
srcpath = "/home/zlinc/caffe-deconvnet-master/python-demo/selectImg/"
for image in sorted(os.listdir(srcpath)):
	feat_vis(srcpath+image)


# This function saves the resized 227*227 RGB images
def get_resized(image):
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    out = net.forward()
    image_name = image.split('/')[-1]
    plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    # save the resized image, make sure that the following directory exists
    plt.savefig("resized/"+image_name[:18]+"_resized.png", dpi = 400, bbox_inches='tight', transparent=True)
	
srcpath = "/home/mshduan/caffe-deconvnet-master/python-demo/selectImg/"
for image in sorted(os.listdir(srcpath)):
    get_resized(srcpath+image)	


