"""Law of Propagation of Uncertainties retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput
from curepy.container.retrieval_result import RetrievalResult

from scipy.optimize import minimize
import numpy as np
import comet_maths as cm
from functools import partial

class LPU(BaseRetrieval):
    """LPU retrieval object"""
    
    def __init__(
        self,
        Jx = None
    ):

        self.Jx = Jx

    def run_retrieval(self, 
                      retrieval_input: RetrievalInput,
                      return_corr: bool = True):
        
        self.retrieval_input = retrieval_input
        
        self._check_retrieval_input()

        theta_0 = self.generate_theta_0(self.retrieval_input.measurement_function_obj.initial_guess)
        
        res = minimize(self.minimiser,
                       theta_0)
        
        if self.Jx is None:
            Jx = self.calculate_Jx(res.x)
        else:
            Jx = self.Jx

        u_func, corr_x = self.process_inverse_jacobian(Jx, res.x)
        
        return RetrievalResult(x = res.x,
                               u_x = u_func,
                               corr_x = corr_x if return_corr else None)
        
    def process_inverse_jacobian(self, J, x):
        covx = self.calculate_measurand_covariance(x, J, self.retrieval_input.measurement_obj.invcov,
                                                   Sa_inv = self.retrieval_input.prior_obj.Sa_inv)
        u_func = np.sqrt(np.diag(covx))
        corr_x = cm.convert_cov_to_corr(covx, u_func)
        
        return u_func, corr_x

    def calculate_measurand_covariance(self, x, J, Sy_inv, Sa_inv=None, Sb_inv=None):
        
        if Sy_inv is not None and Sb_inv is not None:
            Se_inv = Sy_inv + Sb_inv
          
        elif Sy_inv is not None and Sb_inv is None:
            Sb = self.retrieval_input.ancillary_obj.calculate_b_cov()
            if Sb is not None:
                Jb = self.calculate_Jb(x)
                Sb_y = np.dot(np.dot(Jb, Sb), Jb.T)
                Se = np.linalg.inv(Sy_inv) + Sb_y 
            else:
                Se = np.linalg.inv(Sy_inv)
        
            Se_inv = np.linalg.inv(Se)

        else:
            raise ValueError("Covariance must be set for LPU method")
            
        if Sa_inv is not None:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J) + Sa_inv)
        else:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J))
        
    def calculate_Jx(self, x):
        
        meas_func_fixed_b = partial(self.retrieval_input.measurement_function_obj.measurement_function_flattened_output,
                                    b = self.retrieval_input.ancillary_obj.b)
        
        Jx = cm.calculate_Jacobian(meas_func_fixed_b, x)
        
        return Jx
    
    def calculate_Jb(self, x):
        
        b_flat = np.hstack(self.retrieval_input.ancillary_obj.b)
        b_shape_list = [b.shape for b in self.retrieval_input.ancillary_obj.b]
        
        meas_func_fixed_x = lambda b: self.retrieval_input.measurement_function_obj.measurement_function_flattened_b(x, b, b_shape_list)
        
        Jb = cm.calculate_Jacobian(meas_func_fixed_x, b_flat)
        
        return Jb