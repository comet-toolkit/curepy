"""Container for Prior information"""

import numpy as np

implemented_prior_shapes = {"uniform": [],#todo: populate dict
                            }

class Prior:
    def __init__(self,
        shape: str = None,
        params: dict = {}
    ):
        
        self._check_inputs(shape, params)
        #todo: add function to initialise prior function
            
    def _check_inputs(shape, params):
        #check shape
        if shape.lower() not in implemented_prior_shapes:
            raise ValueError(
                f"The provided prior shape ({shape}) is not an implemented prior distribution shape {implemented_prior_shapes.keys()}"
            )
        #check required params exist 
        if all(params.keys()) in implemented_prior_shapes[shape]:# todo: fix check and possibly add optional param check
            raise ValueError(
                f"Prior shape ({shape}) requires the following inputs in param dictionary {implemented_prior_shapes[shape]}"
            )
        
        
        
        
    
        
        