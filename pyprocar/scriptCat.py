from .utilsprocar import UtilsProcar



def cat(inFiles,outFile,gz=False):
    """
    This module concatenates multiple PROCARs.
    """
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
