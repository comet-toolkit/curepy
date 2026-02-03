"""Container for Prior information"""

import numpy as np
from curepy.utilities.distributions import *

implemented_prior_shapes = {"uniform": {"function": ln_uniform,
                                        "params": ["minimum", "maximum"]}
                            }

class Prior:
    def __init__(self,
        prior_shape: str = None,
        prior_params: dict = {}
    ):
        
        self._check_inputs(prior_shape, prior_params)
        self.lnprior = implemented_prior_shapes[prior_shape]["function"]
        self.prior_params = prior_params
    
    @staticmethod   
    def _check_inputs(shape, params):
        #check shape
        if shape.lower() not in implemented_prior_shapes:
            raise ValueError(
                f"The provided prior shape ({shape}) is not an implemented prior distribution shape {implemented_prior_shapes.keys()}"
            )
        #check required params exist 
        if sorted(list((params.keys()))) != sorted(implemented_prior_shapes[shape]["params"]):
            raise ValueError(
                f"Prior shape ({shape}) requires the following inputs in param dictionary {implemented_prior_shapes[shape]}"
            )
    
        
        