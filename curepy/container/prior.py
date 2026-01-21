"""Container for Prior information"""

import numpy as np

class Prior:
    def __init__(self,
        shape: str = None,
        params: dict = {}
    ):
        self.shape = shape
        
        
        