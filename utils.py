import yaml
from importlib import import_module 
import setup

class DictDotNotation(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self

def load_yaml(conf_path):
    with open(conf_path) as file:
        config = yaml.safe_load(file.read())
    return DictDotNotation(config)

def formater_config(config_path):
    module_name = config_path.replace("/", ".")
    module_name = module_name.replace(".py", "")
    config  = import_module(module_name)
    return config

