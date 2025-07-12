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

logger = logging.getLogger(__name__)






def get_orthonormal_basis(normal):
    if np.abs(np.dot(normal, [0, 0, 1])) < 0.99:
        v_temp = np.array([0, 0, 1])  # Not parallel to normal
    else:
        v_temp = np.array([0, 1, 0])  # Not parallel to normal
        
    u = np.cross(v_temp,normal).astype(np.float32)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u).astype(np.float32)
    v /= np.linalg.norm(v)  # Ensure normalization
    
    return u, v

def transform_points_to_uv(
                           points:np.ndarray, 
                           u: np.ndarray, 
                           v: np.ndarray, 
                           origin:np.ndarray = np.array([0, 0, 0]),
                           ):
    points_shifted = points - origin
    return np.column_stack(
        [np.dot(points_shifted, u), np.dot(points_shifted, v)]
    )

def find_plane_limits(plane_points:np.ndarray):
    u_limits = plane_points[:, 0].min(), plane_points[:, 0].max()
    v_limits = plane_points[:, 1].min(), plane_points[:, 1].max()
    return u_limits, v_limits

def get_uv_grid(
    grid_interpolation:tuple[int, int],
    u_limits:tuple[float, float],
    v_limits:tuple[float, float],
    ):
    grid_u, grid_v = np.mgrid[
        u_limits[0] : u_limits[1] : complex(0, grid_interpolation[0]),
        v_limits[0] : v_limits[1] : complex(0, grid_interpolation[1]),
    ]
    return grid_u, grid_v

def get_uv_grid_points(grid_u:np.ndarray, grid_v:np.ndarray):
    return np.vstack([grid_u.ravel(), grid_v.ravel()]).T

def get_uv_grid_kpoints(origin:np.ndarray, uv_points:np.ndarray, uv_transformation_matrix:np.ndarray = None, u:np.ndarray = None, v:np.ndarray = None, normal:np.ndarray = None):
    if uv_transformation_matrix is None:
        uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    return origin + uv_points @ uv_transformation_matrix

def get_uv_transformation_matrix(u:np.ndarray, v:np.ndarray):
    return np.vstack([u, v])

def get_transformation_matrix(u:np.ndarray, v:np.ndarray, normal:np.ndarray):
    uv_transformation_matrix = get_uv_transformation_matrix(u, v)
    transformation_matrix = np.vstack([uv_transformation_matrix, normal])
    return transformation_matrix



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