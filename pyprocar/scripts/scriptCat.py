__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"


import glob
import os
import re
from typing import List

import numpy as np

from ..io.abinit import AbinitOutput
from ..utils import UtilsProcar, welcome


def cat(
    inFiles: List[str] = None,
    outFile: str = "PROCAR_merged",
    gz: bool = False,
    mergeparallel: bool = False,
    fixformat: bool = False,
    nspin: int = 1,
    abinit_output: str = None,
):
    """
    This module concatenates multiple PROCARs.
    If a list of input PROCAR files is not provided it will merge all the PROCAR_*
    files in the directory.

    Parameters
    ----------
    inFiles : List[str], optional
        A list of PROCAR files to concatenate, by default None
    outFile : str, optional
        String for the output, by default "PROCAR_merged"
    gz : bool, optional
        Boolean if output should be compressed to .gz file, by default False
    mergeparallel : bool, optional
        Boolean for merging PROCARs generated
        from parallel Abinit calculations., by default False
    fixformat : bool, optional
        Boolean to fix formatting issues
        in the Abinit PROCAR file., by default False
    nspin : int, optional
        To detect if the calculation is spin-colinear it checks for the nsppol flag
        in the Abinit output file as set in abinit_output. If not present, set nspin., by default 1
    abinit_output : str, optional
        The abinit output file, by default None
    """

    welcome()

    # reading in all PROCAR_* files and putting it into a list if not provided.
    if inFiles is None:
        inFiles = sorted(glob.glob("PROCAR_*"))
    else:
        pass

    print("Concatenating...")
    print("Input         : ", inFiles)  # ', '.join(inFiles)
    print("Output        : ", outFile)

    if mergeparallel == False and fixformat == False:

        if gz == True:
            print("out compressed: true")

        if gz == True and outFile[-3:] != ".gz":
            outFile += ".gz"
            print(".gz extension appended to the outfile")

        handler = UtilsProcar()
        handler.MergeFiles(inFiles, outFile, gzipOut=gz)
        return

    elif mergeparallel == True and fixformat == False:
        _mergeparallel(inFiles, outFile, nspin, abinit_output)

    elif mergeparallel == True and fixformat == True:
        outFile_temp = "outFile.tmp"
        _mergeparallel(inFiles, outFile_temp, nspin, abinit_output)
        _fixformat(outFile_temp, outFile)
        if os.path.exists(outFile_temp):
            os.remove(outFile_temp)

    elif mergeparallel == False and fixformat == True:
        print("Using fixformat = True without mergeparallel. Input a single PROCAR.")
        _fixformat(inFiles, outFile)


def _mergeparallel(inputfiles=None, outputfile=None, nspin=1, abinit_output=None):
    """This merges Procar files seperated between k-point ranges.
    Happens with parallel Abinit runs.
    """
    print("Merging parallel files...")
    filenames = sorted(inputfiles)

    # creating an instance of the AbinitParser class
    if abinit_output:
        abinitparserobject = AbinitOutput(abinit_output_filepath=abinit_output)
        nspin = int(abinitparserobject.nspin)
    else:
        nspin = int(nspin)

    if nspin != 2:
        with open(outputfile, "w") as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)

    elif nspin == 2:
        # for spin polarized calculations the spin down segments are saved in the
        # second half of the PROCAR's but in reverse k-point order. So we have to
        # fix the order and merge the second half of the PROCAR's.
        spinup_list = filenames[: int(len(filenames) / 2)]
        spindown_list = filenames[int(len(filenames) / 2) :]

        # reading the second line of the header to set as the separating line
        # in the colinear spin PROCAR.
        fp = open(spinup_list[0], "r")
        header1 = fp.readline()
        header2 = fp.readline()
        fp.close()

        # second half of PROCAR files in reverse order.
        spindown_list.reverse()

        # Writing new PROCAR with first spin up, header2 and then
        # spin down (reversed).
        with open(outputfile, "w") as outfile:
            for spinupfile in spinup_list:
                with open(spinupfile) as infile:
                    for line in infile:
                        outfile.write(line)
            outfile.write("\n")
            outfile.write(header2)
            outfile.write("\n")
            for spindownfile in spindown_list:
                with open(spindownfile) as infile:
                    for line in infile:
                        outfile.write(line)


def _fixformat(inputfile=None, outputfile=None):
    """Fixes the formatting of Abinit's Procar
    when the tot projection is not summed and spin directions
    not seperated.
    """
    print("Fixing formatting errors...")
    # removing existing temporary fixed file
    if os.path.exists(outputfile):
        os.remove(outputfile)

    ####### Fixing the parallel PROCARs from Abinit ##########

    rf = open(inputfile, "r")
    data = rf.read()
    rf.close()

    # reading headers
    rffl = open(inputfile, "r")
    first_line = rffl.readline()
    rffl.close()

    # header
    header = re.findall("#\sof\s.*", data)[0]

    # writing to PROCAR
    fp = open(outputfile, "a")
    fp.write(first_line)
    fp.write(str(header) + "\n\n")

    # get all the k-point line headers
    kpoints_raw = re.findall("k-point\s*[0-9]\s*:*.*", data)

    for kpoint_counter in range(len(kpoints_raw)):

        if kpoint_counter == (len(kpoints_raw) - 1):
            # get bands of last k point
            bands_raw = re.findall(
                kpoints_raw[kpoint_counter] + "([a-z0-9\s\n.+#-]*)", data
            )[0]

        else:
            # get bands between k point n and n+1
            bands_raw = re.findall(
                kpoints_raw[kpoint_counter]
                + "([a-z0-9\s\n.+#-]*)"
                + kpoints_raw[kpoint_counter + 1],
                data,
            )[0]

        # get the bands headers for a certain k point
        raw_bands = re.findall("band\s*[0-9]*.*", bands_raw)

        # writing k point header to file
        fp.write(kpoints_raw[kpoint_counter] + "\n\n")

        for band_counter in range(len(raw_bands)):

            if band_counter == (len(raw_bands) - 1):
                # the last band
                single_band = re.findall(
                    raw_bands[band_counter] + "([a-z0-9.+\s\n-]*)", bands_raw
                )[0]

            else:
                # get a single band
                single_band = re.findall(
                    raw_bands[band_counter]
                    + "([a-z0-9.+\s\n-]*)"
                    + raw_bands[band_counter + 1],
                    bands_raw,
                )[0]

            # get the column headers for ion, orbitals and total
            column_header = re.findall("ion\s.*tot", single_band)[0]

            # get number of ions using PROCAR file
            nion_raw = re.findall("#\s*of\s*ions:\s*[0-9]*", data)[0]
            nion = int(nion_raw.split(" ")[-1])

            # the first column of the band. Same as ions
            first_column = []
            for x in single_band.split("\n"):
                if x != "":
                    if x != " ":
                        if x.split()[0] != "ion":
                            first_column.append(x.split()[0])

            # number of spin orientations
            norient = int(len(first_column) / nion)

            # calculate the number of orbital headers (s,p,d etc.)
            for x in single_band.split("\n"):
                if x != "":
                    if x != " ":
                        if x.split()[0] == "ion":
                            norbital = len(x.split()) - 2

            # array to store a single band data as seperate lines
            single_band_lines = []
            for x in single_band.split("\n"):
                if x != "":
                    if x != " ":
                        if x.split()[0] != "ion":
                            single_band_lines.append(x)

            # create empty array to store data (the orbitals + tot)
            bands_orb = np.zeros(shape=(norient, nion, norbital + 1))

            # enter data into bands_orb
            iion = 0
            iorient = 0
            for x in single_band.split("\n"):
                if x != "" and x != " " and x.split()[0] != "ion":
                    iline = x.split()
                    if iion > 1:
                        iion = 0
                        iorient += 1
                    for iorb in range(0, norbital + 1):
                        bands_orb[iorient, iion, iorb] = float(iline[iorb + 1])
                    iion += 1

            # create an array to store the total values
            tot = np.zeros(shape=(norient, norbital + 1))

            # entering data into tot array
            for iorient in range(norient):
                tot[iorient, :] = np.sum(bands_orb[iorient, :, :], axis=0)

            # writing data
            fp.write(raw_bands[band_counter] + "\n\n")
            fp.write(column_header + "\n")

            band_iterator = 0
            total_count = 0
            for orientations_count in range(norient):
                for ions_count in range(nion):
                    fp.write(single_band_lines[band_iterator] + "\n")
                    band_iterator += 1
                fp.write("tot  " + " ".join(map(str, tot[total_count, :])) + "\n\n")
                total_count += 1
    fp.close()
