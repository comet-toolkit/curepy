"""Container class for retrieval results"""
import xarray as xr
import obsarray

class RetrievalResult:
    def __init__(self, x=None, u_x=None, corr_x=None, samples=None, b_samples=None, x_names=None):

        self.values = x
        self.uncertainties = u_x
        self.correlation = corr_x
        self.samples = samples
        self.b_samples = b_samples
        self.x_names = x_names

    def build_obsarray(self,):
        pass