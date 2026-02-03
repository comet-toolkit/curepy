"""Base class for retrieval methods"""

from abc import ABC, abstractmethod
from curepy.container.retrieval_input import RetrievalInput
import numpy as np
from curepy.utilities.maths import lnlike

class BaseRetrieval(ABC):
    """Base retrieval object"""
    
    @abstractmethod
    def run_retrieval(self,
                      retrieval_inputs: RetrievalInput):
        pass

    @staticmethod
    def generate_theta_0(ig):
        
        if hasattr(ig, "__len__"):
            if hasattr(ig[0], "__len__"):
                theta_0 = np.concatenate(ig).flatten()
            else:
                theta_0 = np.array(ig).flatten()
        else:
            theta_0 = np.array([ig])
            
        return theta_0
    
    def _check_retrieval_input(self):
        
        #format and define retrieval input
        if self.retrieval_input.ancillary_obj is None:
            self.retrieval_input.build_ancillary()
        if self.retrieval_input.prior_obj is None:
            self.retrieval_input.build_prior(prior_shape="uniform",
                                        prior_params={"minimum": -np.inf,
                                                      "maximum": np.inf})    
    
    def find_chisum(self,
                    theta,
                    repeat_dims = []):
        
        modelled_data = self.retrieval_input.measurement_function_obj.measurement_function_x(theta, self.retrieval_input.ancillary_obj.b)
        diff = modelled_data - self.retrieval_input.measurement_obj.y#todo: flat?
        if np.isfinite(np.sum(diff)):
            if self.retrieval_input.measurement_obj.invcov is None:
                return np.sum((diff) ** 2 / self.retrieval_input.measurement_obj.u_y**2)
            else:
                if len(repeat_dims) == 0:
                    return np.dot(np.dot(diff.T, self.retrieval_input.measurement_obj.invcov), diff)
                elif len(repeat_dims) == 1:
                    sum = 0
                    for i in range(diff.shape[repeat_dims[0]]):
                        diffi = np.take(diff, i, repeat_dims[0])
                        sum += np.dot(np.dot(diffi.T, self.retrieval_input.measurement_obj.invcov), diffi)
                    return sum
                else:
                    raise ValueError(
                        "Methods for multiple repeat dimensions are not yet implemented,"
                    )
        else:
            print(
                "The difference between model and observations is infinite"
            )
        return np.inf
    
    def lnprob(self, theta):
        lp_prior = self.retrieval_input.prior_obj.lnprior(
            theta,
            **self.retrieval_input.prior_obj.prior_params)
        if not np.isfinite(lp_prior):
            return -np.inf
        
        lp = lnlike(self.find_chisum(theta,
                                     repeat_dims=[]))#todo: placeholder! figure out where to define
        
        return lp_prior + lp
    
    def maximiser(self, theta):
        return self.lnprob(theta)
    
    def minimiser(self, theta):
        return -self.lnprob(theta)
    
    
if __name__ == "__main__":
    pass