import numpy as np
import scipy.optimize as opt
from jensen import Jensen
from topdown import TopDown

class CWBL:

    def __init__(self, D, s, ustar, z0, zh, H, phi, Awf):
        self.jens = Jensen(D, s, ustar, z0, zh, phi)
        self.td = TopDown(zh, z0, np.mean(D), ustar, H)
        self.Awf = Awf

        # Get pie-shaped region
        x,y,u = self.jens.velocity_field(256,256)
        X, Y = np.meshgrid(x,y)
        self.ind = self.__get_pie(X,Y)
        self.ind_turb = self.__get_pie(self.jens.ss[:,0], self.jens.ss[:,1])

    def __fmin_fun(self, kw, Ctp):
        self.jens.kw[:] = np.abs(kw)
        self.jens.calc(Ctp, False)
        x,y,u = self.jens.velocity_field(256,256)
        self.wf = np.sum(self.ind*(u<0.95*self.jens.Uinfty))/np.sum(self.ind)
        Ct = 16*np.mean(Ctp)/(4+np.mean(Ctp))**2
        self.cft = np.pi*Ct*np.mean(self.jens.D)**2*self.jens.N/(4*self.Awf)/self.wf
        self.td.calc(self.cft)
        a = np.mean(Ctp)/(4+np.mean(Ctp))
        P1 = 0.5*self.jens.Uinfty**3*4*a*(1-a)**2
        J = (np.mean(self.jens.P[self.ind_turb])/P1 - self.td.uh**3/self.td.Uinfty**3)**2
        return J

    def __get_pie(self, X, Y):
        # Centroid
        xc = np.mean(self.jens.ss[:,0])
        yc = np.mean(self.jens.ss[:,1])
        # Diameter of wind farm
        Dwf = np.sqrt(4*self.Awf/np.pi)
        # Circle
        ind = (X-xc)**2 + (Y-yc)**2 < (Dwf/2)**2

        # Pie shaped area
        ax = np.cos(self.jens.phi)*(X-xc) + np.sin(self.jens.phi)*(Y-yc)
        ay = -np.sin(self.jens.phi)*(X-xc) + np.cos(self.jens.phi)*(Y-yc)
        self.alpha = np.angle(ax+1j*ay)
        ind = ind*(self.alpha<=np.deg2rad(22.5))*(self.alpha>=np.deg2rad(-22.5))

        # Return indicator
        return ind

    def calc(self, Ctp):
        o = opt.minimize_scalar(self.__fmin_fun, args=(Ctp), options={'xtol': 1e-2})
        # self.__fmin_fun(np.abs(o.x), Ctp)
        self.jens.calc(Ctp, True, np.abs(o.x))
