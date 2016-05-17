import numpy as np
import sys
fout = open(sys.argv[1] + '.normalized', 'w')

def dataToString(label, data):
    datavec = data/np.linalg.norm(data)
    datastr = label + ' ' 
    for i in range(len(datavec)):
        datastr += str(i+1) + ':' + str(datavec[i]) + ' '
    return datastr

with open(sys.argv[1]) as f:
    for line in f:
        lineSpl = line.split()
        vec = []
        for i in range(4096):
			vec.append(float(lineSpl[i+1].split(':')[1]))
        fout.write(dataToString(lineSpl[0],vec) + '\n')
