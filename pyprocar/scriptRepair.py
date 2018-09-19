from .utilsprocar import UtilsProcar




#calls ProcarRepair
def repair(infile,outfile):
 
	print("Input File    : ", infile)
	print("Output File   : ", outfile)

#parsing the file
	handler = UtilsProcar()
	handler.ProcarRepair(infile,outfile)
