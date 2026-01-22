"""Container for Measurement data"""

import numpy as np
import punpy
import comet_maths as cm

class Measurement:
    def __init__(self,
                 y,
                 u_y = None,
                 corr_y = None,
                 ):
        
        self.y = y
        self.u_y = u_y
        self.corr_y = self._format_correlation(corr_y)
        
        self._check_shapes()
        
        if corr_y is not None:
            self.inv_cov = self.calculate_inv_cov(self.u_y, self.corr_y)
        else:
            self.invcov = None
        
    def _check_shapes(self):
        N = len(self.y)
        
        if self.u_y is not None and len(self.u_y) != N:
            raise ValueError("Length of measured variable y must match length of uncertainty variable")
        
        if self.u_y is None and self.corr_y is not None:
            raise ValueError("Uncertainties must be defined if error correlation matrix is defined")
        
        if self.corr_y is not None and self.corr_y.shape[0] != self.corr_y.shape[1]:
            raise ValueError("Error correlation matrix must be a square matrix")
        
        if self.corr_y is not None and self.corr_y.shape[0] != N:
            raise ValueError("Length of measured variable y must match side length of error correlation matrix")       
        
    def _format_correlation(self, corr):

        if corr is None:
            return None
        elif isinstance(corr, str):
            if corr == 'rand':
                return np.eye(len(self.y))
            elif corr == 'syst':
                return np.ones((len(self.y), len(self.y)))
            else:
                raise ValueError('Error correlation matrix must be defined as None, "rand", "syst", or a custom matrix')
        else:
            return corr
        
    def calculate_inv_cov(self, unc, corr):
        
        cov = cm.convert_corr_to_cov(corr, unc)
        
        if np.array_equal(cov, np.diag(np.diag(cov))):
            self.invcov = np.diag(1/np.diag(cov)) 
        else:
            #might need a check for PD here
            self.invcov = np.linalg.inv(cov)
            
              
