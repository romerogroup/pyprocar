# Configuration Management System

This directory contains classes and utilities for managing configurations in our application. We employ the Factory Pattern to dynamically create configuration objects based on specified types.

## Overview of the Factory Pattern

The Factory Pattern is a creational design pattern that provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created. This pattern is particularly useful when you need to manage, generate, or manipulate multiple similar objects which are different in some aspects.

### Benefits of Using the Factory Pattern

- **Flexibility**: The Factory Pattern allows the system to be more flexible by decoupling the creation of objects from their implementation. This makes modifying the system easier as changes to object creation can be made in a single location without affecting the entire codebase.
- **Scalability**: It simplifies adding new variations of objects by just extending the factory to handle them, without disturbing existing code.
- **Maintainability**: Centralizes object creation, making the codebase easier to maintain and debug.

## Implementation in Our System

### `ConfigFactory`

The `ConfigFactory` is a static class that provides a method `create_config`, which takes a `plot_type` and other optional keyword arguments. Depending on the `plot_type`, it returns a configuration object specific to that plot type.

```python
class ConfigFactory:
    @staticmethod
    def create_config(plot_type: PlotType, **kwargs):
        if plot_type == PlotType.FERMI_SURFACE:
            return FermiSurface3DConfig(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
```

### `ConfigManager`

The `ConfigManager` is responsible for merging user-provided configuration settings with the default settings of a configuration object. This allows user preferences to override or extend the predefined settings in a controlled manner.

```python
class ConfigManager:
    @staticmethod
    def merge_configs(default_config: BaseConfig, user_config: Dict[str, Any]):
        for key, value in user_config.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
            else:
                if not hasattr(default_config, 'custom_settings'):
                    default_config.custom_settings = {}
                default_config.custom_settings[key] = value
        return default_config
```

## Example Usage

Here is how you can use `ConfigFactory` and `ConfigManager` in your application to initialize and customize a Fermi surface configuration:

```python
# Create a default configuration for a Fermi surface plot
default_config = ConfigFactory.create_config(PlotType.FERMI_SURFACE)

# User-defined settings that might come from an input form or a settings file
user_config = {
    'surface_opacity': 0.75,
    'background_color': 'black'
}

# Merge the default configuration with user-provided settings
self.config = ConfigManager.merge_configs(default_config, user_config or {})
```

This example demonstrates creating a default configuration for a Fermi surface plot and merging it with additional settings provided by the user, allowing for easy customization of plot appearances.


## Directory Structure

- `base.py`: Contains base classes and utilities used across various configurations.
- `fermi_surface_3d.py`: Definitions for Fermi surface plot configurations.
- `band_structure.py`: (If applicable) Configurations specific to band structure plots.
- `density_of_states.py`: (If applicable) Configurations for density of states plots.
- `__init__.py`: Contains the `ConfigFactory` and `ConfigManager` classes.

For more detailed documentation on each module, refer to the respective module's docstrings or internal documentation.

