class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path

    def load(self):
        import yaml
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
