from .utilsprocar import UtilsProcar
import pyfiglet



#calls ProcarRepair
def repair(infile,outfile):
	"""
	This module calls ProcarRepair to repair the PROCAR file.
	"""

	################ Welcome Text #######################
	print(pyfiglet.figlet_format("PyProcar"))
	print('A Python library for electronic structure pre/post-processing.\n')
	print('Please cite: Herath, U., Tavadze, P., He, X., Bousquet, E., Singh, S., Muñoz, F. & Romero,\
	A., PyProcar: A Python library for electronic structure pre/post-processing.,\
	Computer Physics Communications 107080 (2019).\n')

	#####################################################

	print("Input File    : ", infile)
	print("Output File   : ", outfile)

#parsing the file
	handler = UtilsProcar()
	handler.ProcarRepair(infile,outfile)
