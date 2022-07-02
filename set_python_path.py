"""
Avoids tedious package setup by adding current directory into sys.path
"""

import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
path_to_add = current_path

if path_to_add not in sys.path:
    sys.path.append(path_to_add)