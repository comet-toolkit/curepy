"""Container for ancillary parameter information"""

import numpy as np

class AncillaryParameter():
    def __init__(self,
                 b = None,
                 u_b = None,
                 corr_b = None,
                 corr_between_b = None,
                 b_samples = None,
                 b_iter = None,
                 ):
        
        if b:
            try:
                self.b = np.array(b)
            except:
                self.b = np.array(b, dtype=object)
        else:
            self.b = None
            
        if u_b:
            self.u_b = np.array(u_b)
        else:
            self.u_b = None
            
        if corr_b:
            self.corr_b = np.array(corr_b)
        else:
            self.corr_b = None
            
        if corr_between_b:
            self.corr_between_b = np.array(corr_between_b)
        else:
            self.corr_between_b = None