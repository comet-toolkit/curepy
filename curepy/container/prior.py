"""Container for Prior information"""

import numpy as np
from curepy.utilities.distributions import *
from typing import List
import warnings

implemented_prior_shapes = {"uniform": {"function": ln_uniform,
                                        "params": ["minimum", "maximum"],
                                        "correlation": False},
                            "normal": {"function": ln_normal,
                                       "params": ["mu", "sigma"],
                                       "correlation": True}
                            }

class Prior:
    def __init__(self,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation = None,
    ):
        
        self._check_inputs(prior_shape, prior_params, prior_correlation)
        self.function_list = [implemented_prior_shapes[shape]["function"] for shape in prior_shape]
        self.prior_params = prior_params
        self.lnprior = self.combine_dist_functions
        self.prior_correlation = prior_correlation
    
    @staticmethod   
    def _check_inputs(shape, params, corr):
        #check shape
        if all([shape.lower() not in implemented_prior_shapes for shape in shape]):
            raise ValueError(
                f"A provided prior shape ({shape}) is not an implemented prior distribution shape {implemented_prior_shapes.keys()}"
            )
        #check required params exist 
        for i, sh in enumerate(shape):
            if sorted(list((params[i].keys()))) != sorted(implemented_prior_shapes[sh]["params"]):
                raise ValueError(
                    f"Prior shape ({sh}) requires the following inputs in param dictionary {implemented_prior_shapes[sh]['params']}"
                )
            
            if corr is not None and not implemented_prior_shapes[sh]["correlation"]:
                raise ValueError(
                    f"Prior shape ({sh}) is not compatible with correlation inputs"
                )
            if corr is None and implemented_prior_shapes[sh]["correlation"]:
                warnings.warn("No correlation between priors set, assuming random correlation")
        
    
    def combine_dist_functions(self, xs):
        return lambda: [f(x, **kws) for f, x, kws in zip(self.function_list, xs, self.prior_params)]
        
        