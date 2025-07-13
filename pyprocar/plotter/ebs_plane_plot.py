import logging
import os
from functools import partial
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import Normalize
from pyvista import ColorLike
from pyvista.core.filters import _get_output
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    set_algorithm_input,
)
from scipy.interpolate import LinearNDInterpolator

from pyprocar.plotter.ebs_utils import (
    find_plane_limits,
    get_orthonormal_basis,
    get_transformation_matrix,
    get_uv_grid,
    get_uv_grid_kpoints,
    get_uv_grid_points,
    get_uv_transformation_matrix,
    transform_points_to_uv,
)

logger = logging.getLogger(__name__)

class EBSPlanePlotter:
    
    def __init__(self, ebs_mesh, 
                 normal=(0, 0, 1), 
                 origin=(0, 0, 0), 
                 grid_interpolation=(20, 20), 
                 ax=None, figsize=(8, 7), dpi=100):
        self.ebs_mesh = ebs_mesh
        self.normal = normal
        self.origin = origin
        self.grid_interpolation = grid_interpolation
        
        slice = ebs_mesh.slice(normal=normal, origin=origin)
  
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.fig = ax.get_figure()
            self.ax = ax
    
        self.u, self.v = get_orthonormal_basis(normal=normal)
        self.plane_points = transform_points_to_uv(slice.points, self.u, self.v)
        u_limits, v_limits = find_plane_limits(self.plane_points)

        self.u_grid, self.v_grid = get_uv_grid(grid_interpolation=grid_interpolation,
                                    u_limits=u_limits,
                                    v_limits=v_limits)
        
        self.uv_grid_points = get_uv_grid_points(self.u_grid, self.v_grid)

        self.n_points = self.uv_grid_points.shape[0]
        
    def interpolate_values(self, values:np.ndarray):
        if values.shape[-1] != 3:
            new_values = np.zeros(self.n_points)
            interpolator = LinearNDInterpolator(self.plane_points, values)
            new_values = interpolator(self.uv_grid_points)
        else:
            new_values = np.zeros((self.n_points, values.shape[-1]))
            for icoord in range(values.shape[-1]):
                interpolator = LinearNDInterpolator(self.plane_points, values[..., icoord])
                new_values[..., icoord] = interpolator(self.uv_grid_points)
        return new_values
    
    def points_to_grid(self, points:np.ndarray):
        return points.reshape(self.u_grid.shape)
    
    def project_vector_to_plane(self, vectors:np.ndarray):
        velocity_u = np.dot(vectors, self.u)
        velocity_v = np.dot(vectors, self.v)
        return velocity_u, velocity_v

    def plot_scalars(self, 
                scalars:tuple[str, np.ndarray] | None = None,
                grid_points:np.ndarray | None = None,
                name:str="",
                cmap:str="plasma",
                clim:Tuple[float, float]=None,
                shading:str="gouraud",
                alpha:float=0.7,
                **kwargs):
        self.scalar_name = scalars[0] if scalars is not None else name
        
        if scalars is not None:
            slice = self.ebs_mesh.slice(normal=self.normal, origin=self.origin, scalars=scalars)
            scalars_values = slice.active_scalars
            scalars_grid_points = self.interpolate_values(scalars_values)
        elif grid_points is not None:
            scalars_grid_points = grid_points
        else:
            raise ValueError("Either scalars or grid_points must be provided")
        
        scalars_grid = self.points_to_grid(scalars_grid_points)
        self.scalar_plot = self.ax.pcolormesh(self.u_grid, self.v_grid, scalars_grid, 
                                              shading=shading, 
                                              cmap=cmap, 
                                              alpha=alpha, 
                                              clim=clim,
                                              **kwargs)
   
    def plot_vectors_quiver(self, 
                            vectors:tuple[str, np.ndarray] | None = None,
                            grid_points:np.ndarray | None = None,
                            name:str="",
                            scalar_name:str="",
                            plot_scalar:bool=False,
                            plot_scalar_args:dict=None,
                            angles:str='uv',
                            scale=None,
                            arrow_length_factor:float=1.0,
                            arrow_skip:int=1,
                            scale_units:str='inches',
                            units:str='inches',
                            color=None,
                            cmap: str = "plasma",
                            clim:Tuple[float, float]=None,
                            
                             **kwargs):
        self.vector_name = vectors[0] if vectors is not None else name

        if vectors is not None:
            slice = self.ebs_mesh.slice(normal=self.normal, origin=self.origin, vectors=vectors)
            vectors_values = slice.active_vectors
            vectors_grid_points = self.interpolate_values(vectors_values)
        elif vectors_grid_points is not None:
            vectors_grid_points = grid_points
        else:
            raise ValueError("Either vectors or vectors_grid_points must be provided")

  
        velocity_u, velocity_v = self.project_vector_to_plane(vectors_grid_points)
        magnitude_grid_points = np.sqrt(velocity_u**2 + velocity_v**2)

        grid_u_vec = self.points_to_grid(velocity_u)
        grid_v_vec = self.points_to_grid(velocity_v)
        

        quiver_args=[]
        quiver_args.append(self.u_grid[::arrow_skip, ::arrow_skip])
        quiver_args.append(self.v_grid[::arrow_skip, ::arrow_skip])
        quiver_args.append(grid_u_vec[::arrow_skip, ::arrow_skip])
        quiver_args.append(grid_v_vec[::arrow_skip, ::arrow_skip])
        
        if color is None:
            quiver_args.append(magnitude_grid_points)
            
        if scale is None:
            scale = magnitude_grid_points.max()*3
        scale = scale / arrow_length_factor
        
        cmap = plt.get_cmap(cmap)
        if clim is not None:
            norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
        else:
            norm=plt.Normalize(vmin=magnitude_grid_points.min(), vmax=magnitude_grid_points.max())
        
        
        if plot_scalar and not hasattr(self, "scalar_plot"):
            scalars_name = scalar_name if scalar_name else self.vector_name + "_magnitude"
            plot_scalar_args = plot_scalar_args if plot_scalar_args is not None else {}
            plot_scalar_args["cmap"] = plot_scalar_args.get("cmap", cmap)
            plot_scalar_args["clim"] = plot_scalar_args.get("clim", clim)
            self.plot_scalars(name=scalars_name, 
                              grid_points=magnitude_grid_points,
                              **plot_scalar_args)
            
        self.vector_plot = self.ax.quiver(
            *quiver_args,
            angles=angles,
            scale=scale,
            scale_units=scale_units,
            units = units,
            color=color,
            cmap=cmap,
            norm=norm,
            **kwargs
        )
        
        

    def show_colorbar(self, 
                      show_vectors:bool=False,
                      show_scalars:bool=False,
                      label:str="",
                      vector_label:str="",
                      scalar_label:str="",
                      vector_colorbar_args:dict=None,
                      scalar_colorbar_args:dict=None,
                      **kwargs):
        plot_handles = []
        labels = []
        colorbar_args_list = []
        vector_colorbar_args = vector_colorbar_args if vector_colorbar_args is not None else {}
        scalar_colorbar_args = scalar_colorbar_args if scalar_colorbar_args is not None else {}
        
        
        if show_vectors and show_scalars:
            plot_handles = [self.scalar_plot, self.vector_plot]
            labels = [scalar_label or f"{self.scalar_name}", vector_label or f"{self.vector_name}"]
            colorbar_args_list = []
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(scalar_colorbar_args)
            colorbar_args_list.append(tmp_colorbar_args)
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(vector_colorbar_args)
            colorbar_args_list.append(tmp_colorbar_args)
        elif show_vectors and hasattr(self, "vector_plot"):
            plot_handles = [self.vector_plot]
            labels = [label or f"{self.vector_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(vector_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        elif show_scalars and hasattr(self, "scalar_plot"):
            plot_handles = [self.scalar_plot]
            labels = [label or f"{self.scalar_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(scalar_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        elif not show_vectors and not show_scalars and hasattr(self, "scalar_plot"):
            plot_handles = [self.scalar_plot]
            labels = [label or f"{self.scalar_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(scalar_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        elif not show_vectors and not show_scalars and hasattr(self, "vector_plot"):
            plot_handles = [self.vector_plot]
            labels = [label or f"{self.vector_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(vector_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        else:
            raise ValueError("No plot to show colorbar for")
        
        
        self.colorbars = []
        for plot_handle, label, colorbar_args in zip(plot_handles, labels, colorbar_args_list):
            self.colorbars.append(self.fig.colorbar(plot_handle, label=label, **colorbar_args))
        
    def set_xaxis(self, 
                  label:str='k$_u$ (1/Å)',
                  fontsize:int=12, 
                  **kwargs):
        self.ax.set_xlabel(label, fontsize=fontsize, **kwargs)
        
    def set_yaxis(self, 
                  label:str='k$_v$ (1/Å)',
                  fontsize:int=12, 
                  **kwargs):
        self.ax.set_ylabel(label, fontsize=fontsize, **kwargs)
        
    def set_title(self, 
                  title:str=None, 
                  fontsize:int=12, 
                  **kwargs):
        if title is not None:
            return self.ax.set_title(title, fontsize=fontsize, **kwargs)
        
        if hasattr(self, "scalar_plot"):
            title = f"Scalar {self.scalar_name} Contour Plot"
        elif hasattr(self, "vector_plot"):
            title = f"Vector {self.vector_name} Field Plot"
        elif hasattr(self, "scalar_plot") and hasattr(self, "vector_plot"):
            title=f"Scalar {self.scalar_name} and Vector {self.vector_name} Field Plot"
        else:
            title = ""
        title = title.replace("  ", " ")
        return self.ax.set_title(title, fontsize=fontsize, **kwargs)


    def set_default_params(self):
        self.set_xaxis()
        self.set_yaxis()
        self.set_title()
        
    def show(self, **kwargs):
        plt.show(**kwargs)
        
    def savefig(self, filename:str, **kwargs):
        plt.savefig(filename, **kwargs)
        
    def close(self, **kwargs):
        plt.close(**kwargs)
        
    def __str__(self):
        return f"EBSPlanePlotter(ebs_mesh={self.ebs_mesh})"
    
    def __repr__(self):
        return f"EBSPlanePlotter(ebs_mesh={self.ebs_mesh})"