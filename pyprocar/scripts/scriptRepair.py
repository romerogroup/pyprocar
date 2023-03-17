from ..utils import UtilsProcar
from ..utils import welcome


# calls ProcarRepair
def repair(infile:str, outfile:str):
    """This module calls ProcarRepair to repair the PROCAR file.

    Parameters
    ----------
    infile : str
        The input filename
    outfile : _type_
        The output filename
    """
    welcome()

    print("Input File    : ", infile)
    print("Output File   : ", outfile)

    # parsing the file
    handler = UtilsProcar()
    handler.ProcarRepair(infile, outfile)
