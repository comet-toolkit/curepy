"""Container class for retrieval results"""

class RetrievalResult:
    def __init__(self,
                 x = None,
                 u_x = None,
                 corr_x = None,
                 samples = None,
                 b_samples = None):
        
        self.values = x
        self.uncertainties = u_x
        self.correlation = corr_x
        self.samples = samples
        self.b_samples = b_samples