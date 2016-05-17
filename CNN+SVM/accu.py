
import sys
fin1 = open(sys.argv[1], 'r')
fin2 = open(sys.argv[2], 'r')

lines1 = fin1.readlines()
line_num = len(lines1)
targetList = [int(lines1[x].split()[0]) for x in range(line_num)]

predList = map(lambda x : int(x),fin2.readlines())

correct_num = sum(map( lambda x,y: x==y, predList, targetList))

print "total accuracy: " + str(correct_num) + '/' + str(line_num) + ' = ' + str(float(correct_num)/line_num)

index_bau = filter(lambda x : targetList[x] == 1, range(line_num)) 
index_nothing = filter(lambda x : targetList[x] == -1, range(line_num)) 

correct_num_bau = sum(map( lambda x : predList[x] == 1, index_bau))
print "Baustelle accuracy: " + str(correct_num_bau) + '/' + str(len(index_bau)) + ' = ' + str(float(correct_num_bau)/len(index_bau))

correct_num_ohne_bau = sum(map( lambda x : predList[x] == -1, index_nothing))
print "Ohne Baustelle accuracy: " + str(correct_num_ohne_bau) + '/' + str(len(index_nothing)) + ' = ' + str(float(correct_num_ohne_bau)/len(index_nothing))
