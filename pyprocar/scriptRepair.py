from .utilsprocar import UtilsProcar




#calls ProcarRepair
def repair(infile,outfile):
	"""
	This module calls ProcarRepair to repair the PROCAR file.
	"""
 
	print("Input File    : ", infile)
	print("Output File   : ", outfile)

#parsing the file
	handler = UtilsProcar()
	handler.ProcarRepair(infile,outfile)
