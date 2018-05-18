import os
	
def generate2dkmesh(x1,y1,x2,y2,grid_size):	
	n=grid_size;  #Defines density of kgrid
	delta=abs(x2-x1)/n;

	DATA = open("KGrid.dat","w")
	grid = (n+1)*(n+1)
	# Aqui es el primer cuadrante (x,y)
	DATA.write("square mesh near G STO diagonal orientation\n")
	DATA.write("%d\n" %grid )
	DATA.write("reciprocal\n")

	for kx in range(0,n+1): 
  		x3=x1+kx*delta
  		for ky in range(0,n+1):
  			y3=y1+ky*delta
  			DATA.write(" %12.10f   %12.10f   0.000  1.00000 \n" %(x3,y3) )  ## creating (kx, 0, kz) 2DK- mesh
  
	DATA.close()
