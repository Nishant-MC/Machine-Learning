import numpy as np

def normalize(filename):
	file = open(filename, 'r').readlines()
	X = np.zeros([len(file),len(file[0].split(','))-1])
	y = np.zeros(len(file))
	for i in range(len(file)):
		temp = file[i].strip('\n').split(',')
		X[i] = list(map(int, temp[1:]))
		y[i] = int(temp[0])
	X = 2 * X / 255 - 1

	output = open('normalized_'+filename, 'w')
	output.write('Label'+',')
	for i in range(len(X[0])-1):
		output.write('X'+str(i+1)+',')
	output.write('X'+str(len(X[0]))+'\n')
	for i in range(len(file)):
		output.write(str(y[i])+',')
		for j in range(len(X[i])-1):
			output.write(str(X[i][j])+',')
		output.write(str(X[i][len(X[i])-1])+'\n')
	output.close()

normalize("mnist_train.txt")
normalize("mnist_test.txt")