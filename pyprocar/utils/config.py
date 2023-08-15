import yaml

class ConfigManager:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, file_path: str) -> dict:
        """Load configuration from a YAML file."""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def update_config(self, new_config: dict):
        """Update the current configuration with the provided dictionary."""
        for key,value in new_config.items():
            self.config[key]['value']=value
            
    def get_config(self) -> dict:
        """Retrieve the current configuration."""
        return self.config
