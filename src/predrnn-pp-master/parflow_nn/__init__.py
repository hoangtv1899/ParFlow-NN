__version__ = '0.0.1'

import os.path
from .config import Config
config = Config('parflownn', [os.path.join(os.path.dirname(__file__), 'config.ini')])
