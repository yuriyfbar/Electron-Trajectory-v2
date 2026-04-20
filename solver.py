import os

from eqations import *

print('++++++++++++++++')
if os.path.exists("result_11_equations_EXL_50U_13976_r_0.2_t_0.1_00"
".pkl"):
    print('----------------')
    df = pd.read_pickle('result_11_equations_EXL_50U_13976_r_0.2_t_0.1_00.pkl')
    print(df)
    last_row = df.iloc[-1]
    print(last_row)
    #'pparini','rini','thetini','fiini','pperp2ini','Bpolini','Btotini','Bradini','Btorini','psipolini','psitorini'
    pparini = last_row['pparini']
    rini = last_row['rini']
    thetini = last_row['thetini']
    fiini = last_row['fiini']
    pperp2ini = last_row['pperp2ini']
    Bpolini =  last_row['Bpolini']
    Btotini =  last_row['Btotini']
    Bradini =  last_row['Bradini']
    Btorini =  last_row['Btorini']
    psipolini =  last_row['psipolini']
    psitorini =  last_row['psitorini']
    energyini =  last_row['energyini']
    t_ini = last_row['t_ini']
    thetini=thetini-int(thetini/(2*pi))*2*pi
    fiini=fiini-int(fiini/(2*pi))*2*pi
print('rini=',rini,'thetini=',thetini,'fiini=',fiini,'pparini=',pparini,'energyini=',energyini)
#exit()

num_it=5
nrange=2000
delt=200000
#result_df = pd.DataFrame(columns=['pparnp', 'rnp', 'finp', 'thetnp', 'tnp1',])
result_df = pd.DataFrame(columns=['pparini','rini','thetini','fiini','pperp2ini','Bpolini','Btotini','Bradini','Btorini','psipolini','psitorini','energyini','t_ini',])

import time as timer
from scipy.integrate import odeint,solve_ivp    
for it in range(num_it):
    print(f"num it={it}")
    start_time = timer.time()
    t0c=t_ini
    sf0=spl_q0(t0c)
    sfb=spl_qa(t0c)
    Uloop=spl_U(t0c)
    B0=spl_B(t0c)
    print('t_ini=',t0c,'sf0=',sf0,'sfb=',sfb,'B0=',B0,'Uloop=',Uloop)
    sf=saf_fact(sf0,sfb,rini,a,Uloop)
    print('rini=',rini,'thetini=',thetini,'fiini=',fiini,'pparini=',pparini)

    y0= [pparini,rini,thetini,fiini,pperp2ini,Bpolini,Btotini,Bradini,Btorini,psipolini,psitorini,energyini]
    #y0=[pparini,rini,thetini,fiini]
    time= t_ini + delt  #t1UL
    print('rini=',rini,'thetini=',thetini,'fiini=',fiini,'pparini=',pparini,'energyini=',energyini)
    print('t_ini(s)=',t_ini*R0/ccc*tau_norm,'del_t_calculation(s)=',(time-t_ini)*R0/ccc*tau_norm,'time(s)=',time*R0/ccc*tau_norm)
   
    sol= solve_ivp(fin_fun,
                   [t_ini, time], 
                   y0, 
                   method='DOP853', 
                   t_eval= np.linspace(t_ini, time, nrange), 
                   args=(eqq, m0, ccc, a, R0, delr, delfi, nfi, n, pparini, pperpini, muini),
                   rtol= 1e-7,
                   atol= 1e-10) 
    # print(sol.t)
    # print(len(sol.t))
    t_ini=sol.t[nrange-1]
    pparini=sol.y[0,nrange-1]
    rini=sol.y[1,nrange-1]
    thetini=sol.y[2,nrange-1]
    fiini=sol.y[3,nrange-1]
    pperp2ini=sol.y[4,nrange-1]
    Bpolini=sol.y[5,nrange-1]
    Btotini=sol.y[6,nrange-1]
    Bradini=sol.y[7,nrange-1]
    Btorini=sol.y[8,nrange-1]
    psipolini=sol.y[9,nrange-1]
    psitorini=sol.y[10,nrange-1]
    energyini=sol.y[11,nrange-1]

#    print('thetini=',thetini,'fiini=',fiini)
#    print('int(thetini/(2*pi))*2*pi=',int(thetini/(2*pi))*2*pi,'int(fiini/(2*pi))*2*pi=',int(fiini/(2*pi))*2*pi)
    thetini=thetini-int(thetini/(2*pi))*2*pi
    fiini=fiini-int(fiini/(2*pi))*2*pi


    df = pd.DataFrame({
        'pparini'  : sol.y[0],
        'rini'     : sol.y[1],
        'thetini'  : sol.y[2],
        'fiini'    : sol.y[3],
        'pperp2ini': sol.y[4],
        'Bpolini'  : sol.y[5],
        'Btotini'  : sol.y[6],
        'Bradini'  : sol.y[7],
        'Btorini'  : sol.y[8],
        'psipolini': sol.y[9],
        'psitorini': sol.y[10],
        'energyini': sol.y[11],
        't_ini'    : sol.t
        })
    #print(df.head)
    result_df = pd.concat([result_df, df])
    #print(result_df.head)
    result_df.to_pickle('result_11_equations_EXL_50U_13976_r_0.2_t_0.2_.pkl') 
    print(f"----- Iteration execution time: {(timer.time() - start_time):0.2f} sec -----")
#    df.to_pickle('final_data.pkl') 
#LSODA
#DOP853
# Сохраняем DataFrame в бинарный файл
#result_df.to_pickle('result.pkl') 