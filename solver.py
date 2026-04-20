import os
from logger_config import logger
from eqations import *

#from parameters_FT2_r_3 import *
from parameters import *
t_ini=0.2*ccc_R0/tau_norm
t0c=t_ini
sf0=spl_q0(t0c)
sfb=spl_qa(t0c)
Uloop=spl_U(t0c)
B0=spl_B(t0c)
#print('t_ini=',t0c,'sf0=',sf0,'sfb=',sfb,'B0=',B0,'Uloop=',Uloop)
sf=saf_fact(sf0,sfb,rini,a,Uloop)
R,Btotini,Btorini,Bpolini,Bpol1,Bradini,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitorini,dpsidr,dpsidfi,sf \
=Mag_field(rini,thetini,fiini,R0,a,B0,delfi,nfi,delr,n,sf0,sfb,Uloop)
pperp2ini=pperpini**2    
muini=pperp2ini/Btotini
p2ini=pparini**2+pperp2ini
psipolini=pi*B0*a**2/(sfb-sf0)*log((sf0+(sfb-sf0)*(rini/a)**2)/sf0)
energyini=m01*ccc1**2*(sqrt(1+p2ini)-1)/1.6022e-12

logger.info('+++++++  start  +++++++++')
if os.path.exists("result_11_equations_EXL_50U_13976_r_0.2_t_0.1_00.pkl"):
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

logger.info(f'rini= {rini}, thetini={thetini}, fiini={fiini}, pparini= {pparini}, energyini= {energyini}')
#exit()

num_it=10
nrange=2000
delt=200000
#result_df = pd.DataFrame(columns=['pparnp', 'rnp', 'finp', 'thetnp', 'tnp1',])
columns_list = ['ppar','r','thet','fi','pperp2','Bpol','Btot','Brad','Btor','psipol','psitor','energy','time',]
result_df = pd.DataFrame(columns= columns_list)

import time
from scipy.integrate import odeint,solve_ivp    

t_start = t_ini
for it in range(num_it):
    logger.info(f"   ")
    logger.info(f"----- Iteration {it}. Start ----- ")
    start_time = time.time()
    t0c=t_start
    sf0=spl_q0(t0c)
    sfb=spl_qa(t0c)
    Uloop=spl_U(t0c)
    B0=spl_B(t0c)
    logger.info(f't_start= {t0c}, sf0= {sf0}, sfb={sfb}, B0= {B0}, Uloop= {Uloop}')
    sf=saf_fact(sf0,sfb,rini,a,Uloop)
    logger.info(f'rini= {rini}, thetini= {thetini}, fiini= {fiini}, pparini= {pparini}')

    y0= [pparini, rini, thetini, fiini, pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini]
    t_end= t_start + delt  #t1UL
    logger.info(f'rini= {rini}, thetini= {thetini}, fiini= {fiini}, pparini= {pparini}, energyini= {energyini}')
    logger.info(f't_start(s)= {t_start*R0/ccc*tau_norm}, del_t_calculation(s)= {(t_end-t_start)*R0/ccc*tau_norm}, time(s)={t_end*R0/ccc*tau_norm}')
    logger.info(f'solve_ivp: method= DOP853, t_eval={nrange}')

    sol= solve_ivp(fin_fun,
                   [t_start, t_end], 
                   y0, 
                   method='DOP853', 
                   t_eval= np.linspace(t_start, t_end, nrange), 
                   args=(eqq, m0, ccc, a, R0, delr, delfi, nfi, n, pparini, pperpini, muini),
                   rtol= 1e-7,
                   atol= 1e-10) 
    logger.info(f"Number of function evaluations {sol.nfev}")

    t_start=sol.t[-1]
    y_last = sol.y[:, -1]
    pparini, rini, thetini, fiini, pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini = y_last

    thetini=thetini-int(thetini/(2*pi))*2*pi
    fiini=fiini-int(fiini/(2*pi))*2*pi

    df = pd.DataFrame(sol.y.T, columns=columns_list[0:-1])
    df['time'] =  sol.t

    logger.debug("\n" + df.head().to_string())
    result_df = pd.concat([result_df, df])
    result_df.to_pickle('full_trajectory.pkl') 

    eval_time = time.time() - start_time
    logger.info(f"Number of function evaluations per sec {(sol.nfev/eval_time):0.2f}")
    logger.info(f"----- Iteration {it}. Execution time: {eval_time:0.2f} sec -----")
#    df.to_pickle('final_data.pkl') 
#LSODA
#DOP853
# Сохраняем DataFrame в бинарный файл
#result_df.to_pickle('result.pkl') 