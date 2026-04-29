from cmath import pi
import sys

import numpy as np
import pandas as pd
from numpy import cos,sin
import matplotlib.pyplot as plt    
from config import load_configs
from physical_constants import *

run_cfg = load_configs('discharges/base_shot.toml')
#run_cfg = load_configs('discharges/test_shot.toml')

params = run_cfg.params
ccc_R0= ccc/params.R0
a = params.a
R0 = params.R0
n = params.n

df = pd.read_hdf('results/EXL-50U_13976.h5', 'trajectory')


df['R'] = R0+ df['r']*cos(df['theta'])
df['Z'] = df['r']*sin(df['theta'])
df['time']=df['tau']/ccc_R0*tau_norm
df['floor_phi'] =  np.floor(df['phi']/(2*pi)).astype(int)

# Оставляем только те строки, где текущее значение не равно предыдущему
df_Poincare  = df[df['floor_phi'] != df['floor_phi'].shift()]

print(df.head(20).to_string())
print(f"size= {len(df)}")

print(df_Poincare.head(20).to_string())
print(f"size= {len(df_Poincare)}")
plt.ion() # Включаем интерактивный режим

#plt.figure()
ax = df_Poincare.plot(x= 'R', y='Z', alpha=0.05, edgecolors='none', s=10, kind='scatter', title='Scatter plot')
ax.axis('equal')
plt.draw() # Принудительная отрисовка
plt.pause(0.1)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter( df['theta'], df['r']/a, alpha=0.05, color='blue', edgecolors='none', s=10)
ax.set_rmax(1)
#ax.set_rticks([0.2, 0.4, 0.6, 0.8])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
#ax.set_title("Electron trajectory in poloidal crossection", va='bottom')
#plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.1_segment_4_cross_sect.png')

plt.draw() 
plt.pause(0.1)


plt.figure()
plt.plot(df['time'], df['r']/a, 'b')
plt.title("r(t)/a plot")
plt.xlabel('t(ms)')
plt.ylim(0.,1.0)
#plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
plt.grid()
plt.draw() 
plt.pause(0.1)

plt.figure()
plt.plot(df['time'], df['phi'], 'b')
plt.title("phi(t) plot")
plt.xlabel('t(ms)')
#plt.ylim(0.,1.0)
#plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
plt.grid()
plt.draw() 

plt.pause(0.1)

plt.ioff() # Выключаем интерактивный режим
plt.show() # Блокируем выход, пока вы сами не закроете окна




rpr=df['r']/a
thetpr=df['theta']

mmn=0
mmx= 10000
mmn1=100000
mmx1=109999
mmn2=200000
mmx2=209999
mmn3=300000
mmx3=309990


rpr0=rpr[mmn:mmx]
thetpr0=thetpr[mmn:mmx]
rpr1=rpr[mmn1:mmx1]
thetpr1=thetpr[mmn1:mmx1]
rpr2=rpr[mmn2:mmx2]
thetpr2=thetpr[mmn2:mmx2]
rpr3=rpr[mmn3:mmx3]
thetpr3=thetpr[mmn3:mmx3]
#rpr=rpr[0:629200]
#thetpr=thetpr[0:629200]
#tpr=df['t_ini']
#tpr=tpr[0:45]



sys.exit(0)
#print('rini=',sol[nrange-1,1])
#plt.plot(sol.t, sol.y[1]/a, 'g', label='r(t)/a')
#rpr=df['rini']/a
tinipr=df['time']

#rpr=rpr[mmn:mmx]
tinipr0=tinipr[mmn:mmx]
tinipr1=tinipr[mmn1:mmx1]
tinipr2=tinipr[mmn2:mmx2]
tinipr3=tinipr[mmn3:mmx3]
#plt.plot(df['t_ini'], df['rini']/a, 'g', label='r(t)/a')
plt.figure()
plt.plot(tinipr,rpr, 'm', label='r(t)/a')
plt.plot(tinipr0,rpr0, 'r', label='r(t)/a')
plt.plot(tinipr1,rpr1, 'g', label='r(t)/a')
plt.plot(tinipr2,rpr2, 'y', label='r(t)/a')
plt.plot(tinipr3,rpr3, 'b', label='r(t)/a')
plt.legend(loc='best')
plt.xlabel('t(ms)')
#plt.xlim(0.31,0.32)
#plt.xlim(0.346,0.350)
plt.ylim(0.,1.0)
#plt.xlim(0.348,0.349)
#plt.xlim(0.328,0.331)
plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
plt.grid()
plt.draw() # Принудительная отрисовка
plt.pause(0.1)
#print('final t(ms)=',tinipr[mmx-1:mmx])
#print('final tinipr[mmx]=')

plt.figure()
#print('r[nrange-1]=',sol[nrange-1,1])
#plt.plot(sol.t, sol.y[1], 'g', label='r(t)')
plt.plot(df['time'], df['r'], 'g', label='r(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.xlim(15,47)
#plt.xlim(0.32,0.3351)
#plt.xlim(0,0.0215)
#plt.xlim(0.02,0.022)
plt.ylim(0.0,0.601)
#plt.xlim(47.75,48.1)
#plt.xlim(46.9,46.95)
#plt.xlim(0.52,0.53)
plt.grid()
plt.figure()
plt.draw() # Принудительная отрисовка
plt.pause(0.1)

plt.ioff() # Выключаем интерактивный режим
plt.show() # Блокируем выход, пока вы сами не закроете окна

#plt.plot(sol.t, m01*ccc1**2*(sqrt(1+(sol.y[4])+(sol.y[0])**2)-1.)/1.6022e-12, 'r', label='energy(eV)')
#df['energy']= m01*ccc1**2*(np.sqrt(1+df['pperp2']+(df['ppar'])**2)-1.)/1.6022e-12
enrg=df['energy']
enrg0=enrg[mmn:mmx]
enrg1=enrg[mmn1:mmx1]
enrg2=enrg[mmn2:mmx2]
#plt.plot(df['t_ini'], df['energy'], 'r', label='energy(eV)')
plt.plot(df['time'], df['energy'], 'g', label='Energy(eV)')
plt.plot(tinipr0,enrg0, 'y', label='Energy(eV)')
plt.plot(tinipr1,enrg1, 'r', label='Energy(eV)')
plt.plot(tinipr2,enrg2, 'b', label='Energy(eV)')

#plt.plot(df['t_ini'], df['energyini'], 'g', label='energy(eV)')
plt.legend(loc='best')
plt.xlabel('t(s)')
#plt.xlim(0.508,0.51)
#plt.xlim(0.31,0.32)
#plt.xlim(0.629,0.631)
#plt.xlim(0.221,0.227)
#plt.xlim(0.3490,0.350)
#plt.xlim(0.629,0.630)
#plt.xlim(0.43,0.44)
#plt.xlim(0.436,0.437)
#plt.xlim(0.348,0.349)
#plt.xlim(0.2250,0.2256)
#plt.xlim(0.02,0.022)
#plt.xlim(37.71,37.73)
#plt.ylim(5.7e6,5.8e6)
#plt.xlim(46.5,46.6)
plt.ylim(0.e7,1.e7)
#plt.xlim(52.1,52.15)
#plt.xlim(47.7,48.0)
#plt.xlim(59.8,59.825)
#plt.xlim(63.05,63.1)
#plt.xlim(55.2,55.3)
#plt.xlim(0.0198,0.02)
#plt.ylim(7.5e6,7.7e6)
plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_Wkin.svg')
plt.grid()
plt.show()

df['gamnp']=  np.sqrt(1 + df['pperp2'].astype(float) + df['ppar'].astype(float)**2)
#print(len(pperp2np),len(pparini))
plt.plot(df['time'], df['gamnp'], 'r', label='relativistic gamma factor')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

figure, axes = plt.subplots( 1 )
x= (df['r'].astype(float))*cos(df['thet'].astype(float))/a
y= (df['r'].astype(float))*sin(df['thet'].astype(float))/a
x=x[300000:668000]
y=y[300000:668000]
axes.plot(x, y, 'g', label='r(t)/a')
axes.set_aspect( 1 )

plt.title( 'trajectory' )
plt.show()