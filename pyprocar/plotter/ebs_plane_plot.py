from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


class EBSPlanePlotter:
    
    def __init__(self, ebs_plane, ax=None, figsize=(8, 7), dpi=100):
        self.ebs_plane = ebs_plane
        
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.fig = ax.get_figure()
            self.ax = ax
    
        self.grid_u = ebs_plane.grid_u
        self.grid_v = ebs_plane.grid_v
        
        self.u_coord_max = self.grid_u.max()
        self.v_coord_max = self.grid_v.max()
        self.coord_max = max(self.u_coord_max, self.v_coord_max)
        
        self.n_points = self.grid_u.shape[0] * self.grid_v.shape[0]
        
    
    def plot_scalars(self, 
                        scalars:np.ndarray,
                        name:str="",
                        show_colorbar:bool=True,
                        cmap:str="plasma",
                        clim:Tuple[float, float]=None,
                        shading:str="gouraud",
                        alpha:float=0.7,
                        colorbar_args:dict=None,
                        **kwargs):
        self.scalar_name = name
        scalar_grid = scalars.reshape(self.grid_u.shape)
        
        if scalars.shape[0] != self.n_points:
            raise ValueError("Scalar field must have the same number of points as the points in the grid. \n"
                             "Refer to the grid_resolution attribute of the ElectronicBandStructurePlane object.")
        
        self.scalar_plot = self.ax.pcolormesh(self.grid_u, self.grid_v, scalar_grid, 
                                              shading=shading, 
                                              cmap=cmap, 
                                              alpha=alpha, 
                                              clim=clim,
                                              **kwargs)
        
        
        if show_colorbar:
            colorbar_args = colorbar_args or {}
            colorbar_args["label"] = colorbar_args.get("label", f"Scalar {self.scalar_name}")
            self.colorbar = self.fig.colorbar(self.scalar_plot, **colorbar_args)
        
        
    def plot_vectors_quiver(self, 
                             vectors:np.ndarray,
                             use_magnitude_as_scalar:bool=False,
                             color_arrow_by_scalar:bool=False,
                             plot_scalar_args:dict=None,
                             cmap:str=None,
                             clim:Tuple[float, float]=None,
                             show_colorbar:bool=True,
                             vector_colorbar_args:dict=None,
                             name:str="",
                             angles:str='xy',
                             scale_units:str='xy',
                             arrow_scale:float=1.0,
                             arrow_length=None,
                             arrow_width=0.005,
                             arrow_skip=2,
                             arrow_color="black",
                             zorder=10,
                             **kwargs):
        self.vector_name = name

        if vectors.shape[-1] != 3:
            raise ValueError("Vector field must have 3 components as the last index")
        
        if vectors.shape[0] != self.n_points:
            raise ValueError("Vector field must have the same number of points as the points in the grid. \n"
                             "Refer to the grid_resolution attribute of the ElectronicBandStructurePlane object.")
        
        # vectors_grid = vectors.reshape(self.grid_u.shape)
        
        
        # Project the 3D velocity vectors onto the 2D plane basis
        velocity_u = np.dot(vectors, self.ebs_plane.u)
        velocity_v = np.dot(vectors, self.ebs_plane.v)
        
        # --- 2. Normalize the Vector Field ---
        magnitudes = np.sqrt(velocity_u**2 + velocity_v**2)

        # To avoid division by zero for zero-velocity vectors, we use np.divide
        # which allows us to specify what to do 'where' the condition is false.
        norm_u = np.divide(velocity_u, magnitudes, out=np.zeros_like(velocity_u), where=(magnitudes != 0))
        norm_v = np.divide(velocity_v, magnitudes, out=np.zeros_like(velocity_v), where=(magnitudes != 0))
        
        magnitudes = np.sqrt(velocity_u**2 + velocity_v**2)

        # To avoid division by zero for zero-velocity vectors, we use np.divide
        # which allows us to specify what to do 'where' the condition is false.
        norm_u = np.divide(velocity_u, magnitudes, out=np.zeros_like(velocity_u), where=(magnitudes != 0))
        norm_v = np.divide(velocity_v, magnitudes, out=np.zeros_like(velocity_v), where=(magnitudes != 0))

        # Reshape the normalized vectors for plotting
        norm_u_grid = norm_u.reshape(self.grid_u.shape)
        norm_v_grid = norm_v.reshape(self.grid_u.shape)
        
        
        if use_magnitude_as_scalar:
            plot_scalar_args = plot_scalar_args or {}
            plot_scalar_args["scalars"] = magnitudes
            plot_scalar_args["name"] = f"{name} Magnitude"
            self.plot_scalars(**plot_scalar_args)
        
        if arrow_length is None:
            arrow_length = (1 / (self.coord_max * 0.1))
            
        arrow_length = arrow_length * (1 / arrow_scale)
            
        scalar_grid = magnitudes.reshape(self.grid_u.shape)
        
        
        quiver_args=[]
        quiver_args.append(self.grid_u[::arrow_skip, ::arrow_skip])
        quiver_args.append(self.grid_v[::arrow_skip, ::arrow_skip])
        quiver_args.append(norm_u_grid[::arrow_skip, ::arrow_skip])
        quiver_args.append(norm_v_grid[::arrow_skip, ::arrow_skip])
        
        vector_colorbar_args = vector_colorbar_args or {}
        if arrow_color or kwargs.get("color", None) is not None:
            kwargs["color"] = arrow_color
        
        if color_arrow_by_scalar:
            vector_colorbar_args["label"] = vector_colorbar_args.get("label", f"Vector {self.vector_name} Magnitude")
            quiver_args.append(scalar_grid[::arrow_skip, ::arrow_skip])
            
        if clim is not None:
            norm = Normalize(vmin=clim[0], vmax=clim[1])
        
        self.vector_plot = self.ax.quiver(
            *quiver_args,
            angles=angles,
            scale_units=scale_units,
            scale=arrow_length,
            width=arrow_width,
            zorder=zorder,
            cmap=cmap,
            clim=clim,
            norm=norm,
            **kwargs
        )
        
        
        if show_colorbar and vector_colorbar_args:
            vector_colorbar_args["cmap"] = vector_colorbar_args.get("cmap", "coolwarm")
            self.set_vector_colorbar(**vector_colorbar_args)

        
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
        
        if self.scalar_plot is not None:
            title = f"Scalar {self.scalar_name} Contour Plot"
        elif self.vector_plot is not None:
            title = f"Vector {self.vector_name} Field Plot"
        elif self.scalar_plot is not None and self.vector_plot is not None:
            title=f"Scalar {self.scalar_name} and Vector {self.vector_name} Field Plot"
        else:
            title = ""
        title = title.replace("  ", " ")
        return self.ax.set_title(title, fontsize=fontsize, **kwargs)
    
    def set_scalar_colorbar(self, 
                     label:str="",
                     fontsize:int=12,
                     **kwargs):
        if self.scalar_plot:
            return self.fig.colorbar(self.scalar_plot, label=label, **kwargs)
        
    def set_vector_colorbar(self, 
                     label:str="",
                     **kwargs):
        if self.vector_plot:
            return self.fig.colorbar(self.vector_plot, label=label, **kwargs)
        
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
        return f"EBSPlanePlotter(ebs_plane={self.ebs_plane})"
    
    def __repr__(self):
        return f"EBSPlanePlotter(ebs_plane={self.ebs_plane})"