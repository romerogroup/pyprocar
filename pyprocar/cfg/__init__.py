from typing import Dict, Any

from pyprocar.cfg.base import PlotType, BaseConfig
from pyprocar.cfg.fermi_surface_3d import FermiSurface3DConfig
from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.dos import DensityOfStatesConfig
from pyprocar.cfg.unfold import UnfoldingConfig
from pyprocar.cfg.band_structure_2d import Bandstructure2DConfig
from pyprocar.cfg.fermi_surface_2d import FermiSurface2DConfig
class ConfigFactory:
    """
    Factory class for creating configuration objects based on the plot type.

    This class provides a static method to create the appropriate configuration
    object for different types of plots (Fermi surface, band structure, density of states).

    Methods
    -------
    create_config(plot_type: PlotType, **kwargs) -> BaseConfig
        Creates and returns a configuration object based on the specified 
    Parameters
    ----------
    plot_type : PlotType
        The type of plot for which to create a configuration object.
    **kwargs : dict
        Additional keyword arguments to pass to the configuration object construct
    Returns
    -------
    BaseConfig
        A configuration object of the appropriate type for the specified plo
    Raises
    ------
    ValueError
        If an unknown plot type is specified.
    """

    @staticmethod
    def create_config(plot_type: PlotType, **kwargs):
        if plot_type == PlotType.FERMI_SURFACE_3D:
            return FermiSurface3DConfig(plot_type=plot_type, **kwargs)
        elif plot_type == PlotType.BAND_STRUCTURE:
            return BandStructureConfig(plot_type=plot_type,**kwargs)
        elif plot_type == PlotType.DENSITY_OF_STATES:
            return DensityOfStatesConfig(plot_type=plot_type,**kwargs)
        elif plot_type == PlotType.UNFOLD:
            return UnfoldingConfig(plot_type=plot_type,**kwargs)
        elif plot_type == PlotType.BAND_STRUCTURE_2D:
            return Bandstructure2DConfig(plot_type=plot_type,**kwargs)
        elif plot_type == PlotType.FERMI_SURFACE_2D:
            return FermiSurface2DConfig(plot_type=plot_type,**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        

class ConfigManager:
    """
    Manager class for handling configuration objects.

    This class merges user-provided configuration settings with default configurations.

    Methods
    -------
    merge_configs(default_config: BaseConfig, user_config: Dict[str, Any]) -> BaseConfig
        Updates the `default_config` with settings provided by `user_config`.

    Parameters
    ----------
    default_config : BaseConfig
        The default configuration object to update.
    user_config : Dict[str, Any]
        A dictionary of user-provided configuration settings.

    Returns
    -------
    BaseConfig
        The updated configuration object with settings from both default and user configurations.
    """
    @staticmethod
    def merge_configs(default_config: BaseConfig, user_config: Dict[str, Any]) -> BaseConfig:
        for key, value in user_config.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
            else:
                default_config.custom_settings[key] = value
        return default_config
    
    @staticmethod
    def merge_config(default_config: BaseConfig, attribute: str, value: Any):
        if hasattr(default_config, attribute):
            setattr(default_config, attribute, value)
        else:
            raise AttributeError(f"{attribute} is not a valid configuration option.")
        return default_config
