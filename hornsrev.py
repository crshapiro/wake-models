import numpy as np
import matplotlib.pyplot as plt
from jensen import Jensen
from cwbl import CWBL

def hr_mean(P, Nx, Ny, c):
    """
    Average by rows, including only columns c
    """
    Prow = np.zeros(Nx)
    for i in range(0, Nx):
        ind = c+i*Ny
        Prow[i] = np.mean(P[ind])
    return Prow

# Parameters
Nx = 10
Ny = 8
sx = 7.00
sy = 6.95
D = 80
zh = 70
z0 = 0.002
H = 500
ustar = 0.45
Ct = 0.78
orientation = 270

# Derived values
a = 0.5*(1-np.sqrt(1-Ct))
Ctp = 4*a/(1-a)
Ctp = np.ones(Nx*Ny)*Ctp
col_offset = sy*D*np.sin(np.deg2rad(7))
rot = np.deg2rad(270-orientation)

# Set turbine locations
s = np.zeros((Nx*Ny,2))
for i in range(0, Ny):
    for j in range(0, Nx):
        ij = j*Ny+i
        s[ij,0] = j*sx*D - col_offset*i
        s[ij,1] = i*sy*D

# Calculate using Jensen model
jens = Jensen(D*np.ones(Nx*Ny), s, ustar, z0, zh, rot)
jens.calc(Ctp*np.ones(Nx*Ny))
Pj = hr_mean(jens.P, Nx, Ny, np.arange(1,4))

# Calculating using CWBL model
cw = CWBL(D*np.ones(Nx*Ny), s, ustar, z0, zh, H, rot, sx*Nx*sy*Ny*D**2)
cw.calc(Ctp*np.ones(Nx*Ny))
Pc = hr_mean(cw.jens.P, Nx, Ny, np.arange(1,4))

# Plot
fig = plt.figure(figsize=(5,4))
try:
    A = np.loadtxt('hornsrev/hornsrev%i.csv'%orientation,skiprows=1,delimiter=',')
    plt.plot(A[:,0],A[:,1],'k.-')
except:
    print("No LES avaialble for orientation %i" % orientation)
plt.plot(np.arange(1,Nx+1), Pj/Pj[0], 'r.-')
plt.plot(np.arange(1,Nx+1), Pc/Pc[0], 'b.-')
plt.ylim([0.3,1.01])
plt.xlim([0.8,10.2])
plt.ylabel(r'$P/P_1$')
plt.xlabel(r'Row')
plt.xticks(np.arange(1,Nx,2))
plt.legend(['LES','Jensen', 'CWBL'],frameon=False)
plt.tight_layout()
plt.savefig('hornsrev.pdf')
plt.show()
