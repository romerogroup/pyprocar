import re
import os
import seekpath
import numpy as np

#def kpath(infile,with_time_reversal=True,recipe='hpkot',threshhold=1e-07,symprec=1e-05,angle_tolerence=-1.0):
def kpath(infile):
	file = open(infile,'r')
	POSCAR = file.readlines()
	
	#cell
	cell = POSCAR[2:5]
	array2 = np.zeros(shape=(3,3))

	for i in range(len(cell)):
		array1= np.array(cell[i].split())
		array2[i,:]  = array1.astype(np.float)
	print array2	

	#positions
	positions = POSCAR[7:9]	
	array3 = np.zeros(shape=(np.array(POSCAR[5].split()).astype(np.int)[0],3))

	for j in range(len(positions)):
		array1x= np.array(positions[j].split())
		array3[j,:]  = array1x.astype(np.float)

	print array3	
	

if __name__ == "__main__":
	kpath('POSCAR')


