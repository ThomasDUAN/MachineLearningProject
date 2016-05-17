import os
import sys
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
sys.path.insert(0, '/disk/no_backup/mlprak1/share/scene_recognition/caffe-master/python/')
import caffe

# the number of iterations of fine-tuning
niter = 3000
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('places205CNN_finetune_solver.prototxt')
# initialize the CNN with the paramaters of Places-CNN
solver.net.copy_from('placesCNN/places205CNN_iter_300000.caffemodel')
# for comparison, we also fine-tuned a randomly initialized CNN 
scratch_solver = caffe.SGDSolver('places205CNN_finetune_solver.prototxt')

for it in range(niter):
    scratch_solver.step(1)
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
    if it % 10 == 0:
        print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
print 'done'


plot(np.vstack([train_loss, scratch_train_loss]).T)
plot(np.vstack([train_loss, scratch_train_loss]).clip(0, 4).T)


test_iters = 20
accuracy = 0
scratch_accuracy = 0
for it in arange(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
    scratch_solver.test_nets[0].forward()
    scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters
scratch_accuracy /= test_iters
print 'Accuracy for fine-tuning:', accuracy
print 'Accuracy for training from scratch:', scratch_accuracy

