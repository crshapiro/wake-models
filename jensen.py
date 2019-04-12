import numpy as np
import matplotlib.pyplot as plt

class Jensen:

    def __init__(self, D, s, ustar, z0, zh, phi):
        # Check input arguments
        self.N = np.size(D)
        assert np.size(s,0) == self.N, "s must be size (N,2)"
        assert np.size(s,1) == 2, "s must be size (N,2)"

        # Assign input arguments
        self.D = D
        self.s = s
        self.ustar = ustar
        self.z0 = z0
        self.zh = zh
        self.phi = phi

        # Rotate arrangement of turbines
        self.ss = s
        self.s = np.zeros(np.shape(s))
        self.s[:,0] = np.cos(self.phi)*self.ss[:,0] + np.sin(self.phi)*self.ss[:,1]
        self.s[:,1] = -np.sin(self.phi)*self.ss[:,0] + np.cos(self.phi)*self.ss[:,1]

        # Freestream velocity and wake expansion
        self.kappa = 0.4
        self.Uinfty = self.ustar/self.kappa*np.log(self.zh/self.z0)
        self.kw = self.ustar/self.Uinfty*np.ones(self.N)

        # x-grid for computing deficits
        self.x = np.unique(self.s[:,0])
        self.Nx = np.size(self.x)

        # Grid for disk-averaging
        Nys = 32
        Nzs = 32
        yy, zz = np.meshgrid(np.linspace(-1, 1, Nys), np.linspace(-1, 1, Nzs))
        yy = yy.flatten()
        zz = zz.flatten()
        r = np.sqrt(yy**2 + zz**2)
        yy = yy[r<1]
        zz = zz[r<1]
        self.Ns = np.size(yy)
        self.ys = np.zeros((self.N, self.Ns))
        self.zs = np.zeros((self.N, self.Ns))
        for i in range(0, self.N):
            self.ys[i,:] = yy*0.5*self.D[i]
            self.zs[i,:] = zz*0.5*self.D[i]

        # Initialize other values
        self.Ctp = np.zeros(self.N)
        self.ud = np.ones(self.N)*self.Uinfty
        self.P = np.zeros(self.N)

    def calc(self, Ctp, developing=False, kwi=0.05):
        # check input arguments and assign
        assert np.size(Ctp) == self.N, "Ctp must be size N"
        self.Ctp = Ctp

        # Get velocity deficits
        for j in range(0, self.N):
            u = np.zeros(self.Ns)
            Nw = 0
            for i in np.arange(0, self.N)[self.s[j,0]>self.s[:,0]]:
                Dw = self.D[i] + 2*self.kw[i]*(self.s[j,0]-self.s[i,0])
                a = self.Ctp[i]/(4+self.Ctp[i])
                du = 2*self.Uinfty*a/(Dw/self.D[i])**2
                ind = (self.ys[i,:] + self.s[i,1] - self.s[j,1])**2 +          \
                    (self.zs[i,:])**2  < 0.25*Dw**2
                u[ind] += du**2
                if (np.sum(ind) > 0):
                    Nw+=1

            if developing:
                self.kw[j] = kwi + (self.ustar/self.Uinfty - kwi)*np.exp(-Nw)

            self.ud[j] = (self.Uinfty - np.mean(np.sqrt(u)))                   \
                * (1.0 - self.Ctp[j]/(4+Ctp[j]))

        self.P = 0.5*self.Ctp*self.ud**3

    def velocity_field(self, Nx, Ny):
        # Create grid
        xs = np.linspace(np.min(self.ss[:,0])-7*np.max(self.D),np.max(self.ss[:,0])+7*np.max(self.D),Nx)
        ys = np.linspace(np.min(self.ss[:,1])-7*np.max(self.D),np.max(self.ss[:,1])+7*np.max(self.D),Ny)
        XS, YS = np.meshgrid(xs, ys)
        X = np.cos(self.phi)*XS + np.sin(self.phi)*YS
        Y = -np.sin(self.phi)*XS + np.cos(self.phi)*YS
        u = np.zeros((Ny,Nx))
        for i in range(self.N):
            xx = X - self.s[i,0]
            yy = Y - self.s[i,1]
            Dw = np.maximum(self.D[i] + 2*self.kw[i]*(xx), self.D[i])
            a = self.Ctp[i]/(4+self.Ctp[i])
            ind = (xx>0)*(np.abs(yy) < 0.5*Dw)
            du = 2*self.Uinfty*a/(Dw/self.D[i])**2*ind
            u += du**2

        u = self.Uinfty - np.sqrt(u)

        return xs, ys, u
