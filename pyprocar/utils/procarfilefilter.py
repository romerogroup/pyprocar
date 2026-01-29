from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from pyprocar.utils.utilsprocar import UtilsProcar


class ProcarFileFilter:
    """Process a PROCAR file fields line-wise, specially useful for HUGE
    files. This could be thought as pre-processing, writting a new
    PROCAR-like file but reduced in some way.

    A PROCAR File is basically an multi-dimmensional arrays of data, the
    largest being:
    spd_data[#kpoints][#band][#ispin][#atom][#orbital]

    while the number of Kpoints d'ont seems a target for reduction
    (omission or averaging), the other fields can be reduced, for
    instance: grouping the atoms by species or as "surtrate" and
    "adsorbate", or just keeping the bands close to the Fermi energy, or
    discarding the d-orbitals in a s-p system. You got the idea, rigth?

    Example:

    -To group the "s", "p" y "d" orbitals from the file PROCAR and write
     them in PROCAR-spd:

     >>> a = procar.ProcarFileFilter("PROCAR", "PROCAR-new")
     >>> a.FilterOrbitals([[0],[1,2,3],[4,5,6,7,8]], ['s','p', 'd'])

         The PROCAR-new will have just 3+1 columns (orbitals are colum-wise
         , take a look to the file). If you omit the ['s', 'p', 'd'] list,
         the new orbitals will have a generic meaningless name (o1, o2, o3)

    -To group the atoms 1,2,5,6 and 3,4,7,8 from PROCAR and write them
     in PROCAR-new (note the 0-based indexes):

     >>> a = procar.ProcarFileFilter("PROCAR", "PROCAR-new")
     >>> a.FilterAtoms([[0,1,4,5],[2,3,6,7]])

     -To select just the total density (ie: ignoring the spin-resolved stuff,
      if any)from PROCAR and write it in PROCAR-new:

     >>> a = procar.ProcarFileFilter("PROCAR", "PROCAR-new")
     >>> a.FilterSpin([0])

    """

    infile: str | None
    outfile: str | None
    log: logging.Logger
    ch: logging.Handler

    def __init__(
        self,
        infile: str | None = None,
        outfile: str | None = None,
        loglevel: int = logging.DEBUG,
    ) -> None:
        """Initialize the class.

        Params: `infile=None`, input fileName
        """
        self.infile = infile
        self.outfile = outfile

        # We want a logging to tell us what is happening
        self.log = logging.getLogger("ProcarFileFilter")
        self.log.setLevel(loglevel)
        # This is a handler for logging, by now just keep it
        # untouched. Dont really matters its usage
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(logging.Formatter("%(name)s::%(levelname)s: %(message)s"))
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)
        # At last, one message to the logger.
        self.log.debug("ProcarFileFilter instanciated")

    def setInFile(self, infile: str) -> None:
        """Sets a input file `infile`, it can contains the path to the file"""
        self.infile = infile
        self.log.info("Input File: " + infile)

    def setOutFile(self, outfile: str) -> None:
        """Sets a output file `outfile`, it can contains the path to the file"""
        self.outfile = outfile
        self.log.info("Out File: " + outfile)

    def FilterOrbitals(
        self,
        orbitals: list[list[int]],
        orbitalsNames: list[str],
    ) -> None:
        """
        Reads the file already set by SetInFile() and writes a new
        file already set by SetOutFile(). The new file only has the
        selected/grouped orbitals.

        Args:

        -orbitals: nested iterable with the orbitals indexes to be
          considered. For example: [[0],[2]] means select the first
          orbital ("s") and the second one ("pz").
          [[0],[1,2,3],[4,5,6,7,8]] is ["s", "p", "d"].

        -orbitalsNames: The name to be put in each new orbital field (of a
          orbital line). For example ["s","p","d"] is a good
          `orbitalsName` for the `orbitals`=[[0],[1,2,3],[4,5,6,7,8]].
          However, ["foo", "bar", "baz"] is equally valid.

        Note:
          -The atom index is not counted as the first field.
          -The last column ('tot') is so important that it is always
           included. Do not needs to be called
        """
        assert self.infile is not None
        assert self.outfile is not None
        # setting iostuff, this method -and class- should not made any
        # checking about IO, that is the job of the caller
        self.log.info("In File: " + self.infile)
        self.log.info("Out File: " + self.outfile)
        # open the files
        fout = open(self.outfile, "w")
        fopener = UtilsProcar()
        fin = fopener.OpenFile(self.infile)
        for line in fin:
            if re.match(r"\s*ion\s*", line):
                # self.log.debug("orbital line found: " + line)
                line = " ".join(["ion"] + orbitalsNames + ["tot"]) + "\n"

            elif re.match(r"\s*\d+\s*", line) or re.match(r"\s*tot\s*", line):
                # self.log.debug("data line found: " + line)
                fields = line.split()
                # all floats to an array
                data = np.array(fields[1:], dtype=float)
                # setting a new line, keeping just the first value
                new_fields: list[str | np.floating[Any]] = list(fields[:1])
                for orbset in orbitals:
                    new_fields.append(data[orbset].sum())
                # the last value ("tot") always  should be written
                new_fields.append(data[-1])
                # converting to str
                str_fields = [str(x) for x in new_fields]
                line = " ".join(str_fields) + "\n"
            fout.write(line)


    def FilterAtoms(self, atomsGroups: list[list[int] | int]) -> None:
        """
        Reads the file already set by SetInFile() and writes a new
        file already set by SetOutFile(). The new file only has the
        selected/grouped atoms.

        Args:

        -atomsGroups: nested iterable with the atoms indexes (0-based) to
          be considered. For example: [[0],[2]] means select the first and
          the second atoms. While [[1,2,3],[4,5,6,7,8]] means select the
          contribution of atoms 1+2+3 and 4+5+6+7+8

        Note:
          -The atom index is c-based (or python) beginning with 0
          -The output has a dummy atom index, without any intrisic meaning

        """
        # if the user forgot the [...], i.e. [0,[1,2]], we need to add them
        normalized: list[list[int]] = []
        for item in atomsGroups:
            if isinstance(item, int):
                normalized.append([item])
            else:
                normalized.append(item)

        assert self.infile is not None
        assert self.outfile is not None
        # setting iostuff, this method -and class- should not made any
        # checking about IO, that is the job of the caller
        self.log.info("In File: " + self.infile)
        self.log.info("Out File: " + self.outfile)
        # open the files
        fout = open(self.outfile, "w")
        fopener = UtilsProcar()
        with fopener.OpenFile(self.infile) as fin:
            # I need to change the numbers of ions, it will needs the second
            # line. The first one is not needed
            fout.write(fin.readline())
            line = fin.readline()
            fields = line.split()
            # the very last value needs to be changed
            fields[-1] = str(len(normalized))
            line = " ".join(fields)
            fout.write(line + "\n")

            # now parsing the rest of the file
            data_lines: list[str] = []
            for line in fin:
                # if line has data just capture it
                if re.match(r"\s*\d+\s*", line):
                    # self.log.debug("atoms line found: " + line)
                    data_lines.append(line)
                # if `line` is a end of th block (begins with 'tot'), do the
                # work. And clean up data then
                elif re.match(r"\s*tot\s*", line):
                    # self.log.debug("tot line found: " + line)
                    # making an array
                    split_data = [x.split() for x in data_lines]
                    data = np.array(split_data, dtype=float)
                    # iterating on the atoms groups
                    for index in range(len(normalized)):
                        atoms = normalized[index]
                        # summing colum-wise
                        atomLine = data[atoms].sum(axis=0)
                        atom_strs = [str(x) for x in atomLine]
                        # the atom index should not be averaged (anyway now is
                        # meaningless)
                        atom_strs[0] = str(index + 1)
                        atomLineStr = " ".join(atom_strs)
                        fout.write(atomLineStr + "\n")

                    # clean the buffer
                    data_lines = []
                    # and write the `tot` line
                    fout.write(line)
                # otherwise just write this line
                else:
                    fout.write(line)


    def FilterBands(self, Min: int, Max: int) -> None:
        """
        Reads the file already set by SetInFile() and writes a new
        file already set by SetOutFile(). The new file only has the
        selected bands.

        Args:

        -Min, Max:
          the minimum/maximum band  index to be considered, the indexes
          are the same used by vasp (ie written in the file).


        Note: -Since bands are somewhat disordered in vasp you may like to
          consider a large region and made some trial and error

        """
        assert self.infile is not None
        assert self.outfile is not None
        # setting iostuff, this method -and class- should not made any
        # checking about IO, that is the job of the caller
        self.log.info("In File: " + self.infile)
        self.log.info("Out File: " + self.outfile)
        # open the files
        fout = open(self.outfile, "w")
        fopener = UtilsProcar()
        fin = fopener.OpenFile(self.infile)

        # I need to change the numbers of kpoints, it will needs the second
        # line. The first one is not needed
        fout.write(fin.readline())
        line = fin.readline()
        # the third value needs to be changed, however better print it
        self.log.debug("The line contaning bands number is " + line)

        pattern = r"# of bands:\s*(\d+)"
        replacement = "# of bands: " + str(Max - Min + 1)
        line = re.sub(pattern, replacement, line)

        self.log.debug("The new line with # of bands is: " + line)

        fout.write(line + "\n")

        # now parsing the rest of the file
        write = True
        for line in fin:
            if re.match(r"\s*band\s*", line):
                # self.log.debug("bands line found: " + line)
                match = re.match(r"\s*band\s*(\d+)", line)
                assert match is not None
                band = int(match.group(1))
                if band < Min or band > Max:
                    write = False
                else:
                    write = True
            if re.match(r"\s*k-point\s*", line):
                write = True
            if write:
                fout.write(line)

    def FilterSpin(self, components: list[int]) -> None:
        """Reads the file already set by SetInFile() and writes a new
        file already set by SetOutFile(). The new file only has the
        selected part of the density (sigma_i).

        Args:

        -components: For non colinear spin calculations
          the spin component block, for instante [0] menas just
          the density, while [1,2] would be the the sigma_x and sigma_y.
          For colinear spin polarized calculations [0] means spin up
          component and [1] spin down component.


        UPDATE: Colinear spin calculation implemented. It uses regex to
                check for the type of calculation. Hopefully, this won't
                be an issue with memory nowadays.
        """
        assert self.infile is not None
        assert self.outfile is not None
        # setting iostuff, this method -and class- should not made any
        # checking about IO, that is the job of the caller
        self.log.info("In File: " + self.infile)
        self.log.info("Out File: " + self.outfile)

        # First let's see if this is a spin colinear calculation.
        fopener = UtilsProcar()
        fin_test = fopener.OpenFile(self.infile)
        data = fin_test.read()
        colinear_counter = re.findall(r"#\sof\sk-points:", data)

        if len(colinear_counter) == 1:
            # This is a non colinear spin calculation.
            # (Or no spin. But why would anyone want to filter spins in that?)

            print("Non colinear spin calculation detected.")

            # open the files
            fout = open(self.outfile, "w")
            fopener = UtilsProcar()
            with fopener.OpenFile(self.infile) as fin:
                counter = 0
                for line in fin:
                    # if any data found
                    if re.match(r"\s*\d", line):
                        # check if should be written
                        if counter in components:
                            fout.write(line)
                    elif re.match(r"\s*tot", line):
                        if counter in components:
                            fout.write(line)
                        # the next block will belong to other component
                        counter += 1
                    elif re.match(r"\s*ion", line):
                        fout.write(line)
                        counter = 0
                    else:
                        fout.write(line)

        elif len(colinear_counter) == 2:
            # This is a colinear spin calculation.
            # The idea is to seperate spin up [0] and down [1] components
            # and store them in two different PROCARS.
            # They can then be used to plot spin up and spin down
            # bands seperately without plotting spin density or
            # spin magnetization.

            print("Colinear spin calculation detected.")

            # open the files
            fout = open(self.outfile, "w")
            fopener = UtilsProcar()
            spindown_buffer: list[str] = []
            component_counter = 0

            with fopener.OpenFile(self.infile) as fin:
                for line in fin:
                    if re.match(r"PROCAR\s*", line):
                        # First line of the file.
                        spindown_buffer.append(line)

                    if re.match(r"#\sof\sk-points:", line):
                        component_counter += 1

                    if component_counter < 2:
                        if components[0] == 0:
                            # Write to file for spin up component.
                            fout.write(line)
                            # Keep on writing line by line until the start of
                            # the next component, i.e. "# of k-points:".
                    else:
                        # Save spin down component to buffer.
                        # Save later if asked.
                        spindown_buffer.append(line)

            if components[0] == 1:
                fout.writelines(spindown_buffer)


    def FilterKpoints(self, Min: int, Max: int) -> None:
        """
        Reads the file already set by SetInFile() and writes a new
        file already set by SetOutFile(). The new file the
        selected bands.

        Args:

        -Min, Max:
          the minimum/maximum band  kpoint to be considered, the indexes
          are the same used by vasp (i.e. written in the file). Not starting from zero
        """
        assert self.infile is not None
        assert self.outfile is not None
        # setting iostuff, this method -and class- should not made any
        # checking about IO, that is the job of the caller
        self.log.info("In File: " + self.infile)
        self.log.info("Out File: " + self.outfile)
        # open the files
        fout = open(self.outfile, "w")
        fopener = UtilsProcar()
        fin = fopener.OpenFile(self.infile)

        # I need to change the numbers of kpoints, it will needs the second
        # line. The first one is not needed
        fout.write(fin.readline())
        line = fin.readline()
        # the third value needs to be changed, however better print it
        self.log.debug("The line contaning kpoints number is " + line)
        fields = line.split()
        self.log.debug("The number of kpoints is: " + fields[3])
        fields[3] = str(Max - Min + 1)
        line = " ".join(fields)
        fout.write(line + "\n")

        # now parsing the rest of the file
        write = True
        for line in fin:
            if re.match(r"\s*k-point\s*", line):
                self.log.debug("bands line found: " + line)
                match = re.match(r"\s*k-point\s*(\d+)", line)
                assert match is not None
                kpoint = int(match.group(1))
                if kpoint < Min or kpoint > Max:
                    write = False
                else:
                    write = True
            if write:
                fout.write(line)
