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
        self.ln_prior = implemented_prior_shapes[prior_shape]["function"]
        self.prior_params = prior_params
            
    def _check_inputs(shape, params):
        #check shape
        if shape.lower() not in implemented_prior_shapes:
            raise ValueError(
                f"The provided prior shape ({shape}) is not an implemented prior distribution shape {implemented_prior_shapes.keys()}"
            )
        #check required params exist 
        if set(params.keys()) == set(implemented_prior_shapes[shape]["params"]):# todo: fix check and possibly add optional param check
            raise ValueError(
                f"Prior shape ({shape}) requires the following inputs in param dictionary {implemented_prior_shapes[shape]}"
            )
        
        
        
    def lnlike(self, theta):
        # print(theta,self.find_chisum(theta))
        return -0.5 * (self.find_chisum(theta))

    def lnprior(self, theta):
        if np.all(self.downlims.flatten() < theta) and np.all(
            self.uplims.flatten() > theta
        ):
            # if self.syst_uncertainty[0] is None:
            return 0
        # else:
        #     return -0.5*(theta[0]**2/self.syst_uncertainty**2)
        else:
            return -np.inf

    def lnprob(self, theta):
        lp_prior = self.lnprior(theta)
        if not np.isfinite(lp_prior):
            return -np.inf
        lp = self.lnlike(theta)
        return lp_prior + lp
    
        
        