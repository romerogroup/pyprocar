from typing import Dict, Any

from pyprocar.cfg.base import PlotType, BaseConfig
from pyprocar.cfg.fermi_surface_3d import FermiSurface3DConfig
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
        # elif plot_type == PlotType.BAND_STRUCTURE:
        #     return BandStructureConfig(**kwargs)
        # elif plot_type == PlotType.DENSITY_OF_STATES:
        #     return DensityOfStatesConfig(**kwargs)
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
