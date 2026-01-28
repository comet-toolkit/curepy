"""Container for Prior information"""

import numpy as np

implemented_prior_shapes = ["uniform",
                            ]

class Prior:
    def __init__(self,
        shape: str = None,
        params: dict = {}
    ):
        
        self.implemented_prior_shapes = implemented_prior_shapes
        
        if shape.lower() in self.implemented_prior_shapes:
            self.shape = shape
        else:
            raise ValueError(
                f"The provided prior shape ({shape}) is not an implemented prior distribution shape {self.implemented_prior_shapes}"
            )
        
        
    
        
        