import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys


class UtilsProcar:
    """
  This class store handy methods that do not fit any other place
  
  members:

  -Openfile: Tries to open a File, it has suitable values for PROCARs
   and can handle gzipped files
   
  -MergeFiles: concatenate two or more PROCAR files taking care of
   metadata and kpoint indexes. Useful for splitted bandstructures
   calculation.

  -FermiOutcar: it greps the Fermi Energy from a given outcar file.

  -RecLatOutcar: it greps the reciprocal lattice from the outcar.

  """

    def __init__(self, loglevel=logging.WARNING):
        self.log = logging.getLogger("UtilsProcar")
        self.log.setLevel(loglevel)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(
            logging.Formatter("%(name)s::%(levelname)s:"
                              " %(message)s"))
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)
        self.log.debug("UtilsProcar()")
        self.log.debug("UtilsProcar()...done")
        return

###############################SCRIPTS####################################################################

# #calls ProcarRepair
# def scriptRepair(self,infile,outfile):

#   print "Input File    : ", infile
#   print "Output File   : ", outfile

#   #parsing the file
#   handler = UtilsProcar()
#   handler.ProcarRepair(infile,outfile)

#calls MergeFiles
#inFiles should be a list of the PROCAR files that require concatenation
# def scriptCat(self,inFiles,outFile,gz=False):
#   print   "Concatenating:"
#   print   "Input         : ", ', '.join(inFiles)
#   print   "Output        : ", outFile
#   if gz==True:
#       print "out compressed: True"

#   if gz=="True" and outFile[-3:] is not '.gz':
#     outFile += '.gz'
#     print ".gz extension appended to the outFile"

#   handler = UtilsProcar()
#   handler.MergeFiles(inFiles,outFile, gzipOut=gz)
#   return


####################################################################################################################################################

    def OpenFile(self, FileName=None):
        """
    Tries to open a File, it has suitable values for PROCAR and can
    handle gzipped files

    Example: 

    >>> foo =  UtilsProcar.Openfile()
    Tries to open "PROCAR", then "PROCAR.gz"

    >>> foo = UtilsProcar.Openfile("../bar")
    Tries to open "../bar". If it is a directory, it will try to open
    "../bar/PROCAR" and if fails again "../bar/PROCAR.gz"

    >>> foo = UtilsProcar.Openfile("PROCAR-spd.gz")
    Tries to open a gzipped file "PROCAR-spd.gz"

    If unable to open a file, it raises a "IOError" exception.
"""
        import os
        import gzip

        self.log.debug("OpenFile()")
        self.log.debug("Filename :" + FileName)

        if FileName is None:
            FileName = "PROCAR"
            self.log.debug("Input was None, now is: " + FileName)

        #checking if fileName is just a path and needs a "PROCAR to " be
        #appended
        elif os.path.isdir(FileName):
            self.log.info("The filename is a directory")
            if FileName[-1] != r"/":
                FileName += "/"
            FileName += "PROCAR"
            self.log.debug("I will try  to open :" + FileName)

        #checking that the file exist
        if os.path.isfile(FileName):
            self.log.debug("The File does exist")
            #Checking if compressed
            if FileName[-2:] == "gz":
                self.log.info("A gzipped file found")
                inFile = gzip.open(FileName,mode='rt')
            else:
                self.log.debug("A normal file found")
                inFile = open(FileName, "r")
            return inFile

        #otherwise a gzipped version may exist
        elif os.path.isfile(FileName + ".gz"):
            self.log.info(
                "File not found, however a .gz version does exist and will"
                " be used")
            inFile = gzip.open(FileName + ".gz",mode='rt')

        else:
            self.log.debug("File not exist, neither a gzipped version")
            print(FileName)
            raise IOError("File not found")

        self.log.debug("OpenFile()...done")
        return inFile

    def MergeFiles(self, inFiles, outFile, gzipOut=False):
        """
    Concatenate two or more PROCAR files. This methods
    takes care of the k-indexes.

    Useful when large number of K points have been calculated in
    different PROCARs.
    
    Args:
    -inFiles: an iterable with files to be concatenated

    -outFile: a string with the outfile name.

    -gzipOut: whether gzip or not the outout file.

    Warning: spin polarized case is not Ok!

    """
        import gzip

        self.log.debug("MergeFiles()")
        self.log.debug("infiles: " " ,".join(inFiles))

        inFiles = [self.OpenFile(x) for x in inFiles]
        header = [x.readline() for x in inFiles]
        self.log.debug("All the input headers are: \n" + "".join(header))
        metas = [x.readline() for x in inFiles]
        self.log.debug("All the input metalines are:\n " + "".join(metas))
        #parsing metalines

        parsedMeta = [
            list(map(int, re.findall(r"#[^:]+:([^#]+)", x))) for x in metas
        ]
        kpoints = [x[0] for x in parsedMeta]
        bands = set([x[1] for x in parsedMeta])
        ions = set([x[2] for x in parsedMeta])

        #checking that bands and ions match (mind: bands & ions are 'sets'):
        if len(bands) != 1 or len(ions) != 1:
            self.log.error("Number of bands/ions  do not match")
            raise RuntimeError("Files are incompatible")

        newKpoints = np.array(kpoints, dtype=int).sum()
        self.log.info("New number of Kpoints: " + str(newKpoints))
        newMeta = metas[0].replace(str(kpoints[0]), str(newKpoints), 1)
        self.log.debug("New meta line:\n" + newMeta)

        if gzipOut:
            self.log.debug("gzipped output")
            outFile = gzip.open(outFile,mode='wt')
        else:
            self.log.debug("normal output")
            outFile = open(outFile, 'w')
        outFile.write(header[0])
        outFile.write(newMeta)

        #embedded function to change old k-point indexes by the correct
        #ones. The `kreplace.k` syntax is for making the variable 'static'
        def kreplace(matchobj):
            #self.log.debug(print matchobj.group(0))
            kreplace.k += 1
            kreplace.localCounter += 1
            return matchobj.group(0).replace(
                str(kreplace.localCounter), str(kreplace.k))

        kreplace.k = 0
        down = []  #to handle spin-down (if found)
        self.log.debug("Going to replace K-points indexes")
        for inFile in inFiles:
            lines = inFile.read()
            #looking for an extra metada line, if found the file is
            #spin-polarized
            p = re.compile(\
              r'#[\s\w]+k-points:[\s\d]+#[\s\w]+bands:[\s\d]+#[\w\s]+ions:\s*\d+\s*')
            lines = p.split(lines)
            up = lines[0]
            if len(lines) == 2:
                down.append(lines[1])
                self.log.info("Spin-polarized PROCAR!")
            #closing inFile
            inFile.close()
            kreplace.localCounter = 0
            up = re.sub('(\s+k-point\s*\d+\s*:)', kreplace, up)
            outFile.write(up)

        #handling the spin-down channel, if present
        if down:
            self.log.debug("writing spin down metadata")
            outFile.write("\n")
            outFile.write(newMeta)
        kreplace.k = 0
        for group in down:
            kreplace.localCounter = 0
            group = re.sub('(\s+k-point\s*\d+\s*:)', kreplace, group)
            outFile.write(group)

        self.log.debug("Closing output file")
        outFile.close()
        self.log.debug("MergeFiles()...done")
        return

    def FermiOutcar(self, filename):
        """Just finds all E-fermi fields in the outcar file and keeps the
    last one (if more than one found).

    Args:
    -filename: the file name of the outcar to be readed

    """
        self.log.debug("FermiOutcar(): ...")
        self.log.debug("Input filename : " + filename)

        outcar = open(filename, "r").read()
        match = re.findall(r"E-fermi\s*:\s*(-?\d+.\d+)", outcar)[-1]
        self.log.info("Fermi Energy found : " + match)
        self.log.debug("FermiOutcar(): ...Done")
        return float(match)

    def RecLatOutcar(self, filename):
        """Finds and return the reciprocal lattice vectors, if more than
    one set present, it return just the last one.

    Args: 
    -filename: the name of the outcar file  to be read
    
    """
        self.log.debug("RecLatOutcar(): ...")
        self.log.debug("Input filename : " + filename)

        outcar = open(filename, "r").read()
        #just keeping the last component
        recLat = re.findall(r"reciprocal\s*lattice\s*vectors\s*([-.\s\d]*)",
                            outcar)[-1]
        self.log.debug("the match is : " + recLat)
        recLat = recLat.split()
        recLat = np.array(recLat, dtype=float)
        #up to now I have, both direct and rec. lattices (3+3=6 columns)
        recLat.shape = (3, 6)
        recLat = recLat[:, 3:]
        self.log.info("Reciprocal Lattice found :\n" + str(recLat))
        self.log.debug("RecLatOutcar(): ...Done")
        return recLat

    def ProcarRepair(self, infilename, outfilename):
        """It Tries to repair some stupid problems due the stupid fixed
    format of the stupid fortran.

    Up to now it only separes k-points as the following:
    k-point    61 :    0.00000000-0.50000000 0.00000000 ...
    to
    k-point    61 :    0.00000000 -0.50000000 0.00000000 ...

    But as I found new stupid errors they should be fixed here.
    """
        self.log.debug("ProcarRepair(): ...")
        infile = self.OpenFile(infilename)
        fileStr = infile.read()
        infile.close()

        # Fixing bands issues (when there are more than 999 bands)
        # band *** # energy    6.49554019 # occ.  0.00000000
        fileStr = re.sub(r'(band\s)(\*\*\*)', r'\1 1000', fileStr)

        # Fixing k-point issues
        fileStr = re.sub(r'(\.\d{8})(\d{2}\.)', r'\1 \2', fileStr)
        fileStr = re.sub(r'(\d)-(\d)', r'\1 -\2', fileStr)

        fileStr = re.sub(r'\*+', r' -10.0000 ', fileStr)

        outfile = open(outfilename, 'w')
        outfile.write(fileStr)
        outfile.close()

        self.log.debug("ProcarRepair(): ...Done")
        return
