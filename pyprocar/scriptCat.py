from .utilsprocar import UtilsProcar
import pyfiglet


def cat(inFiles,outFile,gz=False):
    """
    This module concatenates multiple PROCARs.
    """
      ################ Welcome Text #######################
    print(pyfiglet.figlet_format("PyProcar"))
    print('A Python library for electronic structure pre/post-processing.\n')
    print('Please cite: Herath, U., Tavadze, P., He, X., Bousquet, E., Singh, S., Muñoz, F. & Romero,\
    A., PyProcar: A Python library for electronic structure pre/post-processing.,\
    Computer Physics Communications 107080 (2019).\n')

    #####################################################

    print("Concatenating:")
    print("Input         : ", inFiles)    # ', '.join(inFiles)
    print("Output        : ", outFile)
    if gz==True:
        print("out compressed: True")
    
    if gz=="True" and outFile[-3:] is not '.gz':
      outFile += '.gz'
      print(".gz extension appended to the outFile")
    
    handler = UtilsProcar()
    handler.MergeFiles(inFiles,outFile, gzipOut=gz)
    return
