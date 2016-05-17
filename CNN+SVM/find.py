import sys
fin1 = open(sys.argv[1], 'r') # target file
fin2 = open(sys.argv[2], 'r') # prediction file
fin3 = open(sys.argv[3], 'r') # yes 
fin4 = open(sys.argv[4], 'r') # no 


lines1 = fin1.readlines()
line_num = len(lines1)
targetList = [int(lines1[x].split()[0]) for x in range(line_num)]

predList = map(lambda x : int(x),fin2.readlines())

index_bau = filter(lambda x : targetList[x] == 1, range(line_num))
index_nothing = filter(lambda x : targetList[x] == -1, range(line_num))

index_wrong_bau = filter( lambda x : predList[x] != 1, index_bau)
index_wrong_nothing = filter( lambda x : predList[x] != -1, index_nothing)

index_right_bau = filter( lambda x : predList[x] == 1, index_bau)
index_right_nothing = filter( lambda x : predList[x] == -1, index_nothing)

feature_wrong_bau = map(lambda x : lines1[x].split(' ',1)[1], index_wrong_bau)
feature_wrong_nothing = map(lambda x : lines1[x].split(' ',1)[1], index_wrong_nothing)

feature_right_bau = map(lambda x : lines1[x].split(' ',1)[1], index_right_bau)
feature_right_nothing = map(lambda x : lines1[x].split(' ',1)[1], index_right_nothing)

wrong_list_bau = []
right_list_bau = []
lines = fin3.readlines()
i = 1
for line in lines:
    if line.split(' ',1)[1] in feature_wrong_bau:
        wrong_list_bau.append(i)
    elif line.split(' ',1)[1] in feature_right_bau:
        right_list_bau.append(i)	
    i += 1
    
wrong_list_nothing = []
right_list_nothing = []
lines = fin4.readlines()
i = 1
for line in lines:
    if line.split(' ',1)[1] in feature_wrong_nothing:
        wrong_list_nothing.append(i)
    elif line.split(' ',1)[1] in feature_right_nothing:
        right_list_nothing.append(i)
    i += 1   

print "roadworks: "
print wrong_list_bau
print  right_list_bau 
print "non_roadworks: "
print wrong_list_nothing
print right_list_nothing 
