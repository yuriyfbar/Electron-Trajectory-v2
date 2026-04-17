import numpy as np
from scipy.interpolate import CubicSpline
from parameters_EXL_50U_13976 import *


t0U=0     #22  # ms,
t=t0U
t1UL=1
print('del_t=',t1UL-t0U)
print('t=',t,'t0U=',t0U,'t1UL=',t1UL)
t0U=t0U/tau_norm*ccc_R0
t1UL=t1UL/tau_norm*ccc_R0


print('after normalisation')
print('t0U=',t0U,'t1UL=',t1UL)
print('del_t_norm=',t1UL-t0U)
krange=23
tnp=np.linspace(t0U, t1UL, krange)
tnp1=np.linspace(t0U, t1UL, 2*krange)
#Uloopnp=np.linspace(0, 1, nrange) 


def Uloop_t(x,t0U,t1UL):
    x=(x-t0U)/(t1UL-t0U)
    Uloop = -7.887*x**6 + 80.237*x**5 - 144.31*x**4 + 90.855*x**3 - 21.72*x**2 + 2.0764*x + 0.6478
    return(Uloop)
x=0
x=x/tau_norm*ccc_R0
z=Uloop_t(x,t0U,t1UL)
print('x*tau_norm/ccc_R0',x,'z=Uloop_t(x,t0U,t1UL)=',z)
x=0.1
x1=(x*tau_norm/ccc_R0 )/(59.024-13.43)
#y = -1E-06*x**5 + 0.0002*x**4 - 0.0188*x**3 + 0.7412*x**2 - 14.063*x + 106.24
y1 =  -7.887*x**6 + 80.237*x**5 - 144.31*x**4 + 90.855*x**3 - 21.72*x**2 + 2.0764*x + 0.6478
print('x1',x1,'z=Uloop_t(x1,t0U,t1UL)=',y1)
#x=59.041
#z=Uloop_t(x)
#print('x',x,'z=Uloop_t(x)=',z)
#exit()
Uloopnp=Uloop_t(tnp,t0U,t1UL)
spl_U=CubicSpline(tnp,Uloopnp)
import matplotlib.pyplot as plt


plt.plot(tnp*tau_norm/ccc_R0, Uloopnp, 'g', label='Uloop')
plt.plot(tnp1*tau_norm/ccc_R0, spl_U(tnp1), 'b', label='Uloop_spline')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#exit()

def B0_t(x,t0U,t1UL):
    x=(x-t0U)/(t1UL-t0U)
    #B0 = 0.0002*x**3 - 0.0443*x**2 + 1.9616*x - 3.0964
    B0 = 0.71
    return(B0) 
#tnp=np.linspace(t0B, t7B, nrange)

B0np=np.linspace(0, 1, krange)

for i in range(krange):
    B0np[i]=B0_t(tnp[i],t0U,t1UL)


spl_B=CubicSpline(tnp,B0np)
der_spl_B=spl_B.derivative()

plt.plot(tnp*tau_norm/ccc_R0,B0np, 'g', label='B0')
plt.plot(tnp1*tau_norm/ccc_R0, spl_B(tnp1), 'b', label='B0_spline')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#exit()

def Cur_t(x,t0U,t1UL):
    x=(x-t0U)/(t1UL-t0U)
    cur = 45249*x**6 - 134815*x**5 + 148661*x**4 - 73887*x**3 + 13833*x**2 + 905.74*x + 44.505
    return(cur)   
curnp=np.linspace(0, 1, krange)

for i in range(krange):
    curnp[i]=Cur_t(tnp[i],t0U,t1UL)
spl_cur=CubicSpline(tnp,curnp)
#sys.exit()
plt.plot(tnp*tau_norm/ccc_R0, curnp, 'g', label='Ip')
plt.plot(tnp1*tau_norm/ccc_R0, spl_cur(tnp1), 'b', label='Ip_spline')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#exit()


Bpl_curnp=np.linspace(0, 1,10*krange)
#Bpl_curnp=2*1.e-7*curnp*1.e3/a
Bpl_curnp=2.e-4*spl_cur(tnp1)/a
spl_Bpl=CubicSpline(tnp1,Bpl_curnp)
#sys.exit()
plt.plot(tnp1*tau_norm/ccc_R0, Bpl_curnp, 'g', label='Bpol(a)')
plt.plot(tnp1*tau_norm/ccc_R0, spl_Bpl(tnp1), 'b', label='Bpol(a)_spline')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#exit()
q_anp=abs(a*spl_B(tnp1)/(R0*Bpl_curnp))
spl_qa=CubicSpline(tnp1,q_anp)


#sys.exit()

ctq0=1
t_q0=t0U  
tau_q0=0.03
tau_q10=2.5
t_q0=t_q0  #/tau_norm*ccc_R0
tau_q0=tau_q0/tau_norm*ccc_R0
tau_q10=tau_q10/tau_norm*ccc_R0
q_0_ini=(spl_qa(t_q0)-ctq0)/2

def q0_t(t,t_q0,tau_q0,q_0_ini):
    #print('t-t_q0=',t-t_q0,'t=',t,'t_q0=',t_q0)
    #print('q_0_ini=',q_0_ini,'(t-t_q0)/tau_q0=',(t-t_q0)/tau_q0,'(t-t_q0)/tau_q10=',(t-t_q0)/tau_q10)
    q_0=q_0_ini*(0.85*np.exp(-(t-t_q0)/tau_q0)+0.0*np.exp(-(t-t_q0)/tau_q10)) +ctq0
    return(q_0)
#print('tnp1=',tnp1)
q0np=q0_t(tnp1,t_q0,tau_q0,q_0_ini)
spl_q0=CubicSpline(tnp1,q0np)

#plt.plot(tnp1*tau_norm/ccc_R0, np.log10(q_anp), 'g', label='log10(q(a))')
#plt.plot(tnp1*tau_norm/ccc_R0, np.log10(spl_qa(tnp1)), 'b', label='log10(q(a)_spline)')
#plt.plot(tnp1*tau_norm/ccc_R0, np.log10(spl_q0(tnp1)), 'g', label='log10(q(0)_spline)')
plt.plot(tnp1*tau_norm/ccc_R0, (q_anp), 'r', label='(q(a))')
plt.plot(tnp1*tau_norm/ccc_R0, (spl_qa(tnp1)), 'b', label='(q(a)_spline)')
plt.plot(tnp1*tau_norm/ccc_R0, (spl_q0(tnp1)), 'g', label='(q(0)_spline)')
plt.ylim(0,10)
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
