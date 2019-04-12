import numpy as np
import matplotlib.pyplot as plt

class TopDown:

    def __init__(self, zh, z0, D, ustar, H):
        self.zh = zh
        self.z0 = z0
        self.D = D
        self.ustar = ustar
        self.H = H

        self.R = self.D*0.5
        self.kappa = 0.4
        self.z0_lo = self.z0
        self.Uinfty = self.ustar/self.kappa*np.log(self.zh/self.z0)

    def calc(self, cft):
        self.cft = cft
        self.nu_w_star = 28*np.sqrt(0.5*cft)
        beta = self.nu_w_star/(1+self.nu_w_star)
        self.z0_hi = self.zh*(1+self.R/self.zh)**beta*np.exp(-(0.5*cft         \
            / self.kappa**2 + (np.log(self.zh/self.z0_lo                       \
            *(1-self.R/self.zh)**beta))**-2)**-0.5)
        self.ustar_hi = self.ustar*np.log(self.H/self.z0_lo)                   \
            / np.log(self.H/self.z0_hi)
        self.ustar_lo = self.ustar_hi*np.log(self.zh/self.z0_hi                \
            * (1+self.R/self.zh)**beta)                                        \
            / np.log(self.zh/self.z0_lo*(1-self.R/self.zh)**beta)
        self.uh = self.ustar_hi/self.kappa*np.log(self.zh/self.z0_hi           \
            * (1+self.R/self.zh)**beta)
