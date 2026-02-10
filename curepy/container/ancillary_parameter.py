"""Container for ancillary parameter information"""

import numpy as np
import punpy
import comet_maths as cm
import warnings
import curepy.utilities.utilities as util
from scipy.linalg import block_diag

class AncillaryParameter():
    def __init__(self,
                 b = None,
                 u_b = None,
                 corr_b = None,
                 corr_between_b = None,
                 b_samples = None,
                 b_iter = None,
                 ):
        
        self.b = None
        self.u_b = None
        self.corr_b = None
        self.corr_between_b = None
        
        if b is not None:
            self._format_ancillary_data(b, u_b, corr_b, corr_between_b)
            
        self.b_iter = b_iter #todo: rename to make functionality clearer
        self.b_samples = b_samples
    
    def _format_ancillary_data(self, b, u_b, corr_b, corr_between_b):
        multiple_b = False
        if b is not None:
            try:
                self.b = np.array(b)
            except:
                multiple_b = True
                self.b = util.to_ragged_array(b)
                
        if u_b is not None:
            if multiple_b:
                self.u_b = util.to_ragged_array(u_b)
            else:
                self.u_b = np.array(u_b)
                
        if corr_b is not None:
            if multiple_b:
                self.corr_b = util.to_ragged_array([util.format_correlation(self.b[i], corr_b[i]) for i in range(self.b.shape[0])])
            else:
                self.corr_b = util.format_correlation(self.b, corr_b)
                
        if corr_between_b is not None:
            self.corr_between_b = np.array(corr_between_b)
            
            
    def generate_b_samples(self):
        if self.b is None:
            self.b_samples = None
        else:
            prop = punpy.MCPropagation(self.b_iter)
            if self.b_samples is None:
                self.b_samples = prop.generate_MC_sample(
                    self.b, self.u_b, self.corr_b, self.corr_between_b
                )
            else:
                self.b_samples = self.b_samples
                
    def calculate_b_cov(self):
        if self.b is None and self.u_b is None and (self.corr_b is None or self.corr_between_b is None):
            warnings.warn("b, u_b, and a correlation matrix must be defined to calculate a covariance matrix")
            return None
        else:
            if self.corr_b is not None:
                if len(set([len(corr) for corr in self.corr_b])) == 1:
                    total_corr = cm.calculate_flattened_corr(corrs = [corr for corr in self.corr_b],
                                                            corr_between = self.corr_between_b if self.corr_between_b is not None else np.eye(len(self.b)))
                else:
                     total_corr = block_diag(*[corr for corr in self.corr_b])
                     warnings.warn("Correlation matrices for each b have different shapes. Assuming no correlation between different b values.")
                     
                return cm.convert_corr_to_cov(total_corr, np.hstack([b.flatten() for b in self.u_b]))
            else:
                warnings.warn("Correlation matrix must be defined to calculate covariance")
                return None