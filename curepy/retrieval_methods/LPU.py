"""Law of Propagation of Uncertainties retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput
from curepy.container.retrieval_result import RetrievalResult

from scipy.optimize import minimize
import numpy as np
import comet_maths as cm

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

        theta_0 = self.generate_theta_0(self.retrieval_input.measurement_function_obj.initial_guess)
        
        res = minimize(self.find_chisum, theta_0)
        
        if self.Jx is None:
            Jx = res.jac 
            #Jx = cm.calculate_Jacobian(self.retrieval_input.measurement_function_obj.measurement_function, 
             #                          res.x)
        else:
            Jx = self.Jx

        u_func, corr_x = self.process_inverse_jacobian(Jx)
        
        return RetrievalResult(x = res.x,
                               u_x = u_func,
                               corr_x = corr_x if return_corr else None)
        
    def process_inverse_jacobian(self, J):
        covx = self.calculate_measurand_covariance(J, self.retrieval_input.measurement_obj.invcov)
        u_func = np.sqrt(np.diag(covx))
        corr_x = cm.convert_cov_to_corr(covx, u_func)
        
        return u_func, corr_x

    def calculate_measurand_covariance(self, J, Sy_inv, Sa_inv=None, Sb_inv=None):
        if Sy_inv is not None and Sb_inv is not None:
            Se_inv = Sy_inv + Sb_inv
        elif Sy_inv is not None and Sb_inv is None:
            Se_inv = Sy_inv
        else:
            Se_inv = np.linalg.inv(cm.convert_corr_to_cov(
                np.ones((len(self.retrieval_input.measurement_obj.u_y),
                         len(self.retrieval_input.measurement_obj.u_y))),
                self.retrieval_input.measurement_obj.u_y)) 
            
        if Sa_inv:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J) + Sa_inv)
        else:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J))