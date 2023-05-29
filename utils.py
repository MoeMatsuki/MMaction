import yaml
from importlib import import_module 
import setup as setup
import pandas as pd

class DictDotNotation(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self
