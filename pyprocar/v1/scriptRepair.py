from .utilsprocar import UtilsProcar
from .splash import welcome


# calls ProcarRepair
def repair(infile, outfile):
    """
	This module calls ProcarRepair to repair the PROCAR file.
	"""
    welcome()

    print("Input File    : ", infile)
    print("Output File   : ", outfile)

    # parsing the file
    handler = UtilsProcar()
    handler.ProcarRepair(infile, outfile)
