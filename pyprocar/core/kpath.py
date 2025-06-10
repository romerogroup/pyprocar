__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"


import numpy as np
import pyvista

from ..utils import mathematics

# TODO Add fmt option to other codes in write_to_file


class KPath:
    def __init__(
        self,
        knames=None,
        kticks=None,
        special_kpoints=None,
        ngrids=None,
        has_time_reversal=True,
    ):
        """
        The Kpath object to handle labels and ticks for band structure

        Parameters
        ----------
        knames : List, optional
            List of Knames, by default None
        kticks : List, optional
            List of kticks that maps the knames to the kpoints, by default None
        special_kpoints : List, optional
            List of special kpoints, by default None
        ngrids : List, optional
            List of how many points are between special kpoints, by default None
        has_time_reversal : bool, optional
            Determine if the kpoints contain time reversal symmetry, by default True
        """
        latex = "$"
        for x in knames:
            if "$" in x[0] or "$" in x[1]:
                latex = ""

        self.knames = []
        for kname_segment in knames:
            kname_start = kname_segment[0]
            kname_end = kname_segment[1]

            if "gamma" in kname_start.lower():
                kname_start = r"\Gamma"
            if "gamma" in kname_end.lower():
                kname_end = r"\Gamma"

            kname_segment = [latex + kname_start + latex, latex + kname_end + latex]
            self.knames.append(kname_segment)

        self.special_kpoints = special_kpoints
        self.ngrids = ngrids
        self.kticks = kticks
        self.has_time_reversal = has_time_reversal

    def __eq__(self, other):
        knames_equal = self.knames == other.knames
        special_kpoints_equal = np.allclose(self.special_kpoints, other.special_kpoints)
        ngrids_equal = self.ngrids == other.ngrids
        kticks_equal = self.kticks == other.kticks
        has_time_reversal_equal = self.has_time_reversal == other.has_time_reversal

        kpath_equal = (
            knames_equal
            and special_kpoints_equal
            and ngrids_equal
            and kticks_equal
            and has_time_reversal_equal
        )
        return kpath_equal

    @property
    def nsegments(self):
        """The number of band segments

        Returns
        -------
        int
            The number of band segments
        """
        return len(self.knames)

    @property
    def tick_positions(self):
        """The list of tick positions

        Returns
        -------
        List
            The list of tick positions
        """
        if self.kticks is None:
            pos = 0
            tick_positions = [pos]
            for isegment in range(self.nsegments):
                pos += self.ngrids[isegment]
                tick_positions.append(pos - 1)
        else:
            tick_positions = self.kticks
        return tick_positions

    @property
    def tick_names(self):
        """The list of tick names

        Returns
        -------
        List
            The list of tick names
        """
        tick_names = [self.knames[0][0], self.knames[0][1]]
        if len(self.knames) == 1:
            return tick_names
        for isegment in range(1, self.nsegments):
            if self.knames[isegment][0] != self.knames[isegment - 1][1]:
                tick_names[-1] += "|" + self.knames[isegment][0]
            tick_names.append(self.knames[isegment][1])
        return tick_names

    @property
    def kdistances(self):
        """An array with the kdistance along the kpath

        Returns
        -------
        np.ndarray
            An array with the kdistance along the kpath
        """
        distances = []
        for isegment in range(self.nsegments):
            distances.append(
                np.linalg.norm(
                    self.special_kpoints[isegment][0]
                    - self.special_kpoints[isegment][1]
                )
            )
        return np.array(distances)

    def get_optimized_kpoints_transformed(
        self, transformation_matrix, same_grid_size=False
    ):
        """
        A method to get the optimized kpoints after a transformation

        Parameters
        ----------
        transformation_matrix : np.ndarray
            The transformmation matrix.
        same_grid_size : bool
            Boolean to determine if the grid should retain the same size

        Returns
        -------
        pyprocar.core.KPath
            The transformed KPath
        """

        new_special_kpoints = np.dot(self.special_kpoints, transformation_matrix)
        new_ngrids = self.ngrids.copy()
        for isegment in range(self.nsegments):
            kstart = new_special_kpoints[isegment][0]
            kend = new_special_kpoints[isegment][1]
            kpoints_old = np.linspace(
                self.special_kpoints[isegment][0],
                self.special_kpoints[isegment][1],
                self.ngrids[isegment],
            )

            dk_vector_old = kpoints_old[-1] - kpoints_old[-2]
            dk_old = np.linalg.norm(dk_vector_old)

            # this part is to find the direction
            distance = kend - kstart

            # this part is to find the high symmetry points on the path
            expand = (np.linspace(kstart, kend, 1000) * 2).round(0) / 2

            unique_indexes = np.sort(np.unique(expand, return_index=True, axis=0)[1])
            symm_kpoints_path = expand[unique_indexes]

            # this part is to only select poits that are after kstart and not before

            angles = np.array(
                [
                    mathematics.get_angle(x, distance, radians=False)
                    for x in (symm_kpoints_path - kstart)
                ]
            ).round()
            symm_kpoints_path = symm_kpoints_path[angles == 0]
            if len(symm_kpoints_path) < 2:
                continue
            suggested_kstart = symm_kpoints_path[0]
            suggested_kend = symm_kpoints_path[1]

            if np.linalg.norm(distance) > np.linalg.norm(
                suggested_kend - suggested_kstart
            ):
                new_special_kpoints[isegment][0] = suggested_kstart
                new_special_kpoints[isegment][1] = suggested_kend

            # this part is to get the number of gird points in the to have the
            # same spacing is before the transformation
            if same_grid_size:
                new_ngrids[isegment] = int(
                    (
                        np.linalg.norm(
                            new_special_kpoints[isegment][0]
                            - new_special_kpoints[isegment][1]
                        )
                        / dk_old
                    ).round(4)
                    + 1
                )
        return KPath(
            knames=self.knames, special_kpoints=new_special_kpoints, ngrids=new_ngrids
        )

    def get_kpoints_transformed(
        self,
        transformation_matrix,
    ):
        """A method to get the transformed kpoints

        Parameters
        ----------
        transformation_matrix : np.ndarray
            The transformation matrix

        Returns
        -------
        pyprocar.core.KPath
            The transformed KPath
        """
        new_special_kpoints = np.dot(self.special_kpoints, transformation_matrix)
        return KPath(
            knames=self.knames, special_kpoints=new_special_kpoints, ngrids=self.ngrids
        )

    def write_to_file(self, filename="KPOINTS", fmt="vasp"):
        """Write the kpath to a file. Only supports vasp at the moment

        Parameters
        ----------
        filename : str, optional
            _description_, by default "KPOINTS"
        fmt : str, optional
            _description_, by default "vasp"
        """
        with open(filename, "w") as wf:
            if fmt == "vasp":
                wf.write("! Generated by pyprocar\n")
                if len(np.unique(self.ngrids)) == 1:
                    wf.write(str(self.ngrids[0]) + "\n")
                else:
                    wf.write("   ".join([str(x) for x in self.ngrids]) + "\n")
                wf.write("Line-mode\n")
                wf.write("reciprocal\n")
                for isegment in range(self.nsegments):
                    wf.write(
                        " ".join(
                            [
                                "  {:8.4f}".format(x)
                                for x in self.special_kpoints[isegment][0]
                            ]
                        )
                        + "   ! "
                        + self.knames[isegment][0].replace("$", "")
                        + "\n"
                    )
                    wf.write(
                        " ".join(
                            [
                                "  {:8.4f}".format(x)
                                for x in self.special_kpoints[isegment][1]
                            ]
                        )
                        + "   ! "
                        + self.knames[isegment][1].replace("$", "")
                        + "\n"
                    )
                    wf.write("\n")

        return None

    def __str__(self):
        ret = "K-Path\n"
        ret += "------\n"
        for isegment in range(self.nsegments):
            ret += "{:>2}. {:<9}: ({:>.2f} {:>.2f} {:>.2f}) -> {:<9}: ({:>.2f} {:>.2f} {:>.2f})\n".format(
                isegment + 1,
                self.knames[isegment][0],
                self.special_kpoints[isegment][0][0],
                self.special_kpoints[isegment][0][1],
                self.special_kpoints[isegment][0][2],
                self.knames[isegment][1],
                self.special_kpoints[isegment][1][0],
                self.special_kpoints[isegment][1][1],
                self.special_kpoints[isegment][1][2],
            )
        return ret
