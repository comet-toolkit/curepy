"""Container for ancillary parameter information"""

import numpy as np
import punpy

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
            try:
                self.b = np.array(b)
            except:
                self.b = np.array(b, dtype=object)
        if u_b is not None:
            self.u_b = np.array(u_b)
        if corr_b is not None:
            self.corr_b = np.array(corr_b)
        if corr_between_b is not None:
            self.corr_between_b = np.array(corr_between_b)
            
        self.b_iter = b_iter #todo rename to make functionality clearer
        self.b_samples = b_samples
        
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
        