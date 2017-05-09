from numpy import array, dot, random
import numpy as np
import math
import sys

def sigmoid(x):
    return 1.0/(1 + math.exp(-x))

f = open('nntrain.tra', 'r')
lines = f.readlines()
f.close()

f = open('nntest.tra', 'r')
testlines = f.readlines()
f.close()

imagesOf1 = []
imagesOf2 = []
imagesOf3 = []
allimages = []
mapp = []
image = ''

timagesOf1 = []
timagesOf2 = []
timagesOf3 = []
tallimages = []
tmapp = []

for line in lines:
    if len(line.strip('\n')) == 32:
        image += line + '\n'
    else:
        number = line.strip('\n').strip()
        if number == '1' or number == '2' or number == '3':
            tempImage = image.split('\n')
            tempImage = filter(None, tempImage)
            tempImageArray = []
            for i in tempImage:
                arrays = list(i)
                tempImageArray.append(arrays)
            if number == '1':
                imagesOf1.append(tempImageArray)
                mapp.append(1)
            if number == '2':
                imagesOf2.append(tempImageArray)
                mapp.append(2)
            if number == '3':
                imagesOf3.append(tempImageArray)
                mapp.append(3)
            allimages.append(tempImageArray)
            image = ''
        else:
            image = ''

for line in testlines:
    if len(line.strip('\n')) == 32:
        image += line + '\n'
    else:
        number = line.strip('\n').strip()
        if number == '1' or number == '2' or number == '3':
            tempImage = image.split('\n')
            tempImage = filter(None, tempImage)
            tempImageArray = []
            for i in tempImage:
                arrays = list(i)
                tempImageArray.append(arrays)
            if number == '1':
                timagesOf1.append(tempImageArray)
                tmapp.append(1)
            if number == '2':
                timagesOf2.append(tempImageArray)
                tmapp.append(2)
            if number == '3':
                timagesOf3.append(tempImageArray)
                tmapp.append(3)
            tallimages.append(tempImageArray)
            image = ''
        else:
            image = ''
no_of_hidden_units = int(sys.argv[1])
weights = random.rand(65, no_of_hidden_units)
# weights = np.full((65, 10), 1, dtype=float)
# hidden_weights = np.full((10, 3), 1, dtype=float)
hidden_weights = random.rand(no_of_hidden_units,3)
expectedFor1 = array([1, 0, 0])
expectedFor2 = array([0, 1, 0])
expectedFor3 = array([0, 0, 1])
eta = 0.2
printted1 = 0
printted2 = 0
printted3 = 0

for _ in range(500):
    for ppp in range(len(allimages)):
        new_matrix = [[0 for i in range(8)] for j in range(8)]
        rows = 0
        columns = 0
        for i in range(0, 32, 4):
            for j in range(0, 32, 4):
                average = 0
                for k in range(i, i + 4):
                    for l in range(j, j + 4):
                        average += int(allimages[ppp][k][l])
                new_matrix[rows][columns] = average/16.0
                columns += 1
            rows += 1
            columns = 0
        
        newImageArray = array([new_matrix[i][j] for i in range(8) for j in range(8)] + [1])
        netj = dot(newImageArray[np.newaxis], weights)
        # temp = newImageArray
    
        yj = []
        for i in netj:
            for j in range(len(i)):
                yj.append(sigmoid(i[j]))
        
        netk = dot(array(yj), hidden_weights)
    
        zk = []
        for i in netk[np.newaxis]:
            for j in range(len(i)):
                zk.append(sigmoid(i[j]))

        if mapp[ppp]==1:
            tk_zk = expectedFor1 - array(zk)[np.newaxis]
        elif mapp[ppp]==2:
            tk_zk = expectedFor2 - array(zk)[np.newaxis]
        else:
            tk_zk = expectedFor3 - array(zk)[np.newaxis]
        
        fdashofnetk = []
        for i in netk[np.newaxis]:
            for j in range(len(i)):
                fdashofnetk.append(sigmoid(i[j])*(1 - sigmoid(i[j])))
        
        deltak = np.multiply(tk_zk, array(fdashofnetk)[np.newaxis])
        sigma = dot(hidden_weights, deltak.T)
        fdashofnetj = []
        for i in netj:
            for j in range(len(i)):
                fdashofnetj.append(sigmoid(i[j])*(1 - sigmoid(i[j])))
        sumy = dot(hidden_weights, deltak.T)
    
        deltaj = np.multiply(array(fdashofnetj)[np.newaxis], sumy.T)
        
        weights += eta * dot(newImageArray[np.newaxis].T, deltaj)
        hidden_weights += eta * dot(array(yj)[np.newaxis].T, deltak)
        # print zk,mapp[ppp]

total = 0
correct = 0
for ppp in range(len(tallimages)):
    new_matrix = [[0 for i in range(8)] for j in range(8)]
    rows = 0
    columns = 0
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            average = 0
            for k in range(i, i + 4):
                for l in range(j, j + 4):
                    average += int(tallimages[ppp][k][l])
            new_matrix[rows][columns] = average/16.0
            columns += 1
        rows += 1
        columns = 0
    
    newImageArray = array([new_matrix[i][j] for i in range(8) for j in range(8)] + [1])
    netj = dot(newImageArray[np.newaxis], weights)
    # temp = newImageArray

    yj = []
    for i in netj:
        for j in range(len(i)):
            yj.append(sigmoid(i[j]))
    
    netk = dot(array(yj), hidden_weights)

    zk = []
    for i in netk[np.newaxis]:
        for j in range(len(i)):
            zk.append(sigmoid(i[j]))
    # print tmapp[ppp],zk.index(max(zk))+1,zk
    if tmapp[ppp]==1:
        if printted1==0:
            printted1=1
            print tmapp[ppp],zk.index(max(zk))+1,zk,"dsfd",yj
    if tmapp[ppp]==2:
        if printted2==0:
            printted2=1
            print tmapp[ppp],zk.index(max(zk))+1,zk,"dsfd",yj
    if tmapp[ppp]==3:
        if printted3==0:
            printted3=1
            print tmapp[ppp],zk.index(max(zk))+1,zk,"dsfd",yj
    total += 1
    if tmapp[ppp]==(zk.index(max(zk))+1):
        correct += 1
print (correct*100.0)/total
print no_of_hidden_units
for i in weights:
    print ','.join(map(str,list(i)))
# print weights
for i in hidden_weights:
    print ','.join(map(str,list(i)))
# print hidden_weights