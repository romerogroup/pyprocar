__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"


import logging
from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv

from pyprocar.core import BrillouinZone
from pyprocar.utils import mathematics

logger = logging.getLogger(__name__)

def reduced_to_cartesian(kpoints, reciprocal_lattice):
    if reciprocal_lattice is not None:
        return np.dot(kpoints, reciprocal_lattice)
    else:
        print("Please provide a reciprocal lattice when initiating the Procar class")
        return

def format_names(names: List[str], as_latex: bool = False):

    new_names = []
    for x in names:
        
        if x.lower() == "gamma":
            x = r"\Gamma"
            
        if "\\" in x and as_latex:
            x = "$" + x + "$"
        new_names.append(x)
    return new_names

class KPath:
    def __init__(
        self,
        kpoints:np.ndarray=None,
        n_grids:List[int]=None,
        segment_names:List[Tuple[str, str]]=None,
        tick_name_map:Dict[int, str]=None,
        reciprocal_lattice=None,
        discontinuity_threshold=0.2,
        zero_diff_threshold=1e-6,
        as_latex=True,
    ):
        """
        The Kpath object to handle labels and ticks for band structure

        Parameters
        ----------
        kpoints: np.ndarray
            The kpoints to be used for the kpath
        n_grids: List[int]
            The number of grids to be used for each segment
        segment_names: List[Tuple[str, str]]
            This is a list of tuples containing the names of the segments. 
            The first element of the tuple is the name of the start point of the segment 
            and the second element is the name of the end point of the segment.
        tick_name_map: Dict[int, str]
            A dictionary containing the names of ticks on the kpath.
            The key is the index of the tick and the value is the name of the tick.
        reciprocal_lattice: np.ndarray
            The reciprocal lattice of the crystal
        discontinuity_threshold: float
            The threshold for a discontinuity
        zero_diff_threshold: float
            The threshold for a zero difference
        """
        if kpoints is None and n_grids is None:
            raise ValueError("Either kpoints or n_grids must be provided")
        
        self._n_grids = n_grids
        self._kpoints = kpoints
        self.discontinuity_threshold = discontinuity_threshold
        self.zero_diff_threshold = zero_diff_threshold
        self._tick_name_map = tick_name_map
        self._reciprocal_lattice = reciprocal_lattice
        
        
        if self._kpoints is None:
            logger.info("No kpoints provided. Generating kpoints from special kpoints and ngrids")
            self._kpoints = self.generate_points()
            
        
        self._segment_indices, self._continuous_start_indices, self._discontinuity_start_indices = self.get_segment_indices()
        self._segment_names = segment_names
        self._special_kpoint_names = self.get_special_kpoint_names(segment_names=self._segment_names)
        
        self.special_kpoint_names = format_names(self._special_kpoint_names, as_latex=as_latex)
        

    def __eq__(self, other):
        segment_names_equal = self.segment_names == other.segment_names
        special_kpoints_equal = np.allclose(self.special_kpoints, other.special_kpoints)
        n_grids_equal = self.n_grids == other.n_grids
        tick_names_equal = self.tick_names == other.tick_names
        return segment_names_equal and special_kpoints_equal and n_grids_equal and tick_names_equal
    
    def __str__(self):
        ret = "K-Path\n"
        ret += "------\n"
        for isegment, segment_indices in enumerate(self.segment_indices):
            start_name, end_name = self.segment_names[isegment]
            start_kpoint = self.special_kpoint_map[start_name]
            end_kpoint = self.special_kpoint_map[end_name]
            


            ret += "{:>2}. {:<8}: ({:>6.2f} {:>6.2f} {:>6.2f}) -> {:<8}: ({:>6.2f} {:>6.2f} {:>6.2f})\n".format(
                    isegment + 1,
                    start_name,
                    start_kpoint[0],
                    start_kpoint[1],
                    start_kpoint[2],
                    end_name,
                    end_kpoint[0],
                    end_kpoint[1],
                    end_kpoint[2],
                )
            
        ret += "\n"
        ret += "Tick names:    " + "    ".join(f"{name:^8}" for name in self.tick_names) + "\n"
        ret += "Tick positions:" + "    ".join(f"{pos:^8}" for pos in self.tick_positions) + "\n"
        ret += "n_kpoints: " + str(self.n_kpoints) + "\n"
        ret += "n_segments: " + str(self.n_segments) + "\n"
        ret += "n_grids: " + str(self.n_grids) + "\n"
        ret += "discontinuity_indices: " + str(self.discontinuity_start_indices) + "\n"
        ret += "continuous_indices: " + str(self.continuous_start_indices) + "\n"
        
        return ret
    
    @property
    def n_kpoints(self):
        return len(self._kpoints)
    
    @property
    def n_grids(self):
        return self._n_grids
    
    @property
    def n_segments(self):
        """The number of band segments

        Returns
        -------
        int
            The number of band segments
        """
        return len(self.segment_indices)
    
    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice
    
    @property
    def brillouin_zone(self):
        return BrillouinZone(self.reciprocal_lattice, transformation_matrix=[1,1,1])

    @property
    def kpoints(self):
        return self._kpoints
    
    @property
    def k_distances(self):
        return self.get_distances(as_segments=False)
    
    @property
    def segment_indices(self):
        return self._segment_indices
    
    @property
    def knames(self):
        return self.segment_names
    
    @property
    def continuous_start_indices(self):
        return self._continuous_start_indices
    
    @property
    def discontinuity_start_indices(self):
        return self._discontinuity_start_indices

    @property
    def kpoints_cartesian(self):
        return reduced_to_cartesian(self.kpoints, self._reciprocal_lattice)

    @property
    def segment_names(self):
        return self._segment_names
    @segment_names.setter
    def segment_names(self, segment_names):
        if len(segment_names) != self.n_segments:
            raise ValueError(f"Number of segment names must match number of segments. Got {len(segment_names)} names for {self.n_segments} segments")
        self._segment_names = segment_names
        self._special_kpoints = self.get_special_kpoints()
        
    @property
    def special_kpoint_names(self):
        return self._special_kpoint_names
    @special_kpoint_names.setter
    def special_kpoint_names(self, special_kpoint_names):
        if len(special_kpoint_names) != len(self._special_kpoint_names):
            raise ValueError(f"Setting special kpoint names must match the existing number of special kpoint names.\n" 
                             f"Got {len(special_kpoint_names)} special kpoint names for {len(self._special_kpoint_names)} special kpoint names")
        new_segment_names = []
        for i, segment_name_tuple in enumerate(self._segment_names):
            start_name, end_name = segment_name_tuple
            for j, special_kpoint in enumerate(self._special_kpoint_names):
                if special_kpoint == start_name:
                    start_name = special_kpoint_names[j]
                if special_kpoint == end_name:
                    end_name = special_kpoint_names[j]
                     
            new_segment_names.append((start_name, end_name))

        self._segment_names = new_segment_names
        self._special_kpoint_names = special_kpoint_names
        
    @cached_property
    def special_kpoints(self):
        return self.get_special_kpoints(as_segments=True)
    
    @property
    def special_kpoint_map(self):
        special_kpoint_map = {}
        special_kpoints = self.get_special_kpoints(as_segments=False)
        for name, kpoint in zip(self.special_kpoint_names, special_kpoints):
            special_kpoint_map[name] = kpoint
        return special_kpoint_map
       
    def get_special_kpoint_names(self, segment_names: List[Tuple[str, str]] = None):
        if segment_names is None:
            segment_names = self._segment_names
        special_kpoint_names = []
        for i, segment_name in enumerate(segment_names):
            if segment_name[0] not in special_kpoint_names:
                special_kpoint_names.append(segment_name[0])
            if segment_name[1] not in special_kpoint_names:
                special_kpoint_names.append(segment_name[1])
        return special_kpoint_names
    
    def get_special_kpoints(self, as_segments: bool = False, cartesian: bool = False):
        special_kpoints = []
        kpoints = self.kpoints_cartesian if cartesian else self.kpoints
        for segment_indices in self.segment_indices:
            start_kpoint = kpoints[segment_indices[0]]
            end_kpoint = kpoints[segment_indices[-1]]
            
            if as_segments:
                special_kpoints.append((start_kpoint, end_kpoint))
                continue
            
            # Check if start_kpoint is already in the list (using numpy array comparison)
            start_exists = any(np.allclose(start_kpoint, existing) for existing in special_kpoints)
            if not start_exists:
                special_kpoints.append(start_kpoint)
                
            # Check if end_kpoint is already in the list (using numpy array comparison)
            end_exists = any(np.allclose(end_kpoint, existing) for existing in special_kpoints)
            if not end_exists:
                special_kpoints.append(end_kpoint)

        return np.array(special_kpoints)


    @property
    def tick_names_latex(self):
        tick_names_latex = []
        for tick_index, tick_name in enumerate(self.tick_names):
            if "\\" in tick_name:
                tick_name = f"${tick_name}$"
            tick_names_latex.append(tick_name)
        return tick_names_latex

    @property
    def tick_name_map(self):
        """The list of tick names

        Returns
        -------
        List
            The list of tick names
        """
        if self._tick_name_map is None:
            tick_name_map = {}
            for i, segment_indices in enumerate(self._segment_indices):
                start_index = segment_indices[0]
                end_index = segment_indices[-1]
                if i == 0:
                    tick_name_map[start_index] = self._segment_names[i][0]
                    continue
                if i == len(self._segment_indices) - 1:
                    tick_name_map[end_index] = self._segment_names[i][1]
                    continue
                
                if end_index in self.discontinuity_start_indices:
                    tick_name_map[end_index] = self._segment_names[i][0] + "|" + self._segment_names[i][1]
                    
                    # Remove the previous segment end index. To avoid double tick
                    previous_segment_end_index = self._segment_indices[i-1][-1]
                    tick_name_map.pop(previous_segment_end_index)
                elif end_index in self.continuous_start_indices:
                    tick_name_map[end_index] = self._segment_names[i][1]
                else:
                    raise ValueError(f"Segment {i} is not a discontinuity or continuous segment. Likely a bug in get_segment_indices")

            self._tick_name_map = tick_name_map
            
        return self._tick_name_map
    
    @property
    def tick_names(self):
        return list(self.tick_name_map.values())
    
    @property
    def tick_positions(self):
        return list(self.tick_name_map.keys())

    def get_segments(self, isegments: List[int]=None, cartesian: bool = False):
        if isegments is None:
            isegments = list(range(self.n_segments))
            
        kpoints = self.kpoints_cartesian if cartesian else self.kpoints
        
        segments = []
        for segment_indices in self.segment_indices:
            segments.append(kpoints[segment_indices])

        return [segments[i] for i in isegments]

    def get_distances(
        self,
        isegments: List[int] = None,
        as_segments: bool = True,
        cumlative_across_segments: bool = True,
        cartesian: bool = False,
    ):

        segments = self.get_segments(
            isegments=isegments, cartesian=cartesian
        )

        k_segment_distances = []
        previous_segment_max = 0
        for isegment, segment in enumerate(segments):
            k_diffs = np.diff(segment, axis=0)
            k_diffs = np.linalg.norm(k_diffs, axis=1)
            k_distances = np.cumsum(k_diffs)

            k_distances = np.insert(k_distances, 0, 0)
            if cumlative_across_segments:
                k_distances = k_distances + previous_segment_max
                previous_segment_max = k_distances[-1]
         
            k_segment_distances.append(k_distances)

        if as_segments:
            return k_segment_distances
        else:
            return np.concatenate(k_segment_distances)

    def get_segment_indices(self):

        if self._kpoints is None or len(self._kpoints) == 0:
            return np.array([])
            
        # Compute differences between consecutive kpoints
        k_diffs = np.diff(self._kpoints, axis=0)
        
        # Calculate the norm of differences
        k_diff_norms = np.linalg.norm(k_diffs, axis=1)
        
        # Find indices where difference is 0 (or very close to 0)
        continuous_end_indices = list(np.where(k_diff_norms < self.zero_diff_threshold)[0])
        discontinuity_end_indices = list(np.where(k_diff_norms > self.discontinuity_threshold)[0])
        
        print(k_diff_norms[discontinuity_end_indices])
        
        logger.info(f"Continuous indices: {continuous_end_indices}")
        logger.info(f"Discontinuity indices: {discontinuity_end_indices}")
        segment_end_indices = continuous_end_indices + discontinuity_end_indices + [len(self._kpoints) - 1]
        segment_end_indices.sort()
        
        logger.info(f"Found {len(segment_end_indices)} segments")
        
        
        indices = []
        for i, segment_end_index in enumerate(segment_end_indices):
            if i == 0:
                indices.append(np.arange(0, segment_end_indices[i] + 1))
            else:
                indices.append(np.arange(segment_end_indices[i-1] + 1, segment_end_indices[i] + 1))

        return indices, continuous_end_indices, discontinuity_end_indices
    
    def get_continuous_segments(self):
        
        continuous_segments = []
        for isegment, segment_indices in enumerate(self.segment_indices):
            if isegment == 0:
                continuous_segments.append(segment_indices)
                continue

            previous_segment_end_index = self.segment_indices[isegment - 1][-1]
            
            if previous_segment_end_index in self.discontinuity_start_indices:
                continuous_segments.append(segment_indices)
                continue
            
            elif previous_segment_end_index in self.continuous_start_indices:
                continuous_segments[-1] = np.concatenate((continuous_segments[-1], segment_indices))
                continue
            else:
                raise ValueError(f"Segment {isegment} is not a discontinuity or continuous segment. Likely a bug in get_segment_indices")
            
        return continuous_segments
        
    def generate_points(self):
        """
        Generate the kpath points
        """
        kpoints_on_path = None
        for isegment in range(self.n_segments):
            kstart, kend = self.special_kpoints[isegment]
            kpoints = np.linspace(kstart, kend, self.n_grids[isegment])
 
            if len(kpoints_on_path) == 0:
                kpoints_on_path = kpoints
            else:
                kpoints_on_path = np.concatenate((kpoints_on_path, kpoints))
        return kpoints_on_path

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
        new_ngrids = self.n_grids.copy()
        for isegment in range(self.n_segments):
            kstart = new_special_kpoints[isegment][0]
            kend = new_special_kpoints[isegment][1]
            kpoints_old = np.linspace(
                self.special_kpoints[isegment][0],
                self.special_kpoints[isegment][1],
                self.n_grids[isegment],
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
        return KPath(special_kpoints=new_special_kpoints, n_grids=new_ngrids)

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
        return KPath(kpoints=new_special_kpoints, reciprocal_lattice=transformation_matrix)

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
                if len(np.unique(self.n_grids)) == 1:
                    wf.write(str(self.n_grids[0]) + "\n")
                else:
                    wf.write("   ".join([str(x) for x in self.n_grids]) + "\n")
                wf.write("Line-mode\n")
                wf.write("reciprocal\n")
                for isegment in range(self.n_segments):
                    wf.write(
                        " ".join(
                            [
                                "  {:8.4f}".format(x)
                                for x in self.special_kpoints[isegment][0]
                            ]
                        )
                        + "   ! "
                        + self.special_kpoint_names[isegment][0].replace("$", "")
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
                        + self.special_kpoint_names[isegment][1].replace("$", "")
                        + "\n"
                    )
                    wf.write("\n")

        return None
    
    def plot(
        self,
        add_point_labels_args: dict = None,
        bz_add_mesh_args: dict = None,
        as_cartesian: bool = False,
        **kwargs,
    ):
        """
        Plots the band structure.

        """
        add_point_labels_args = add_point_labels_args or {}
        bz_add_mesh_args = bz_add_mesh_args or {}

        p = pv.Plotter()
        
        if as_cartesian:
            kpath = pv.PolyData(self.kpoints_cartesian)
        else:
            kpath = pv.PolyData(self.kpoints)
            
        p.add_mesh(kpath, **kwargs)
        
        special_kpoints = self.get_special_kpoints(as_segments=False, cartesian=as_cartesian)
        special_kpoint_names = self.get_special_kpoint_names()
        p.add_point_labels(
            special_kpoints, special_kpoint_names, **add_point_labels_args
        )

        bz_add_mesh_args["style"] = bz_add_mesh_args.get("style", "wireframe")
        bz_add_mesh_args["line_width"] = bz_add_mesh_args.get("line_width", 2.0)
        bz_add_mesh_args["color"] = bz_add_mesh_args.get("color", "black")
        bz_add_mesh_args["opacity"] = bz_add_mesh_args.get("opacity", 1.0)

        p.add_mesh(
            self.brillouin_zone,
            **bz_add_mesh_args,
        )
        p.show()

        

    
