import gc
import pandas as pd
from config import load_configs, log_config
from logger_config import log_memory_usage, logger
import time
from scipy.integrate import odeint,solve_ivp  

from physical_constants import *
import parameters

run_cfg = load_configs('discharges/base_shot.toml')
log_config(run_cfg)
#parameters.a, parameters.R0, parameters.delr, parameters.delfi, parameters.nfi =  run_cfg

#from parameters import *
# eval const
ccc_R0 = ccc/run_cfg.R0
parameters.ccc_R0 = ccc_R0
parameters.a = run_cfg.a
parameters.R0 = run_cfg.R0
from eqations import *

t_ini=0.2*ccc_R0/tau_norm
t0c=t_ini
sf0=spl_q0(t0c)
sfb=spl_qa(t0c)
Uloop=spl_U(t0c)
B0=spl_B(t0c)
#print('t_ini=',t0c,'sf0=',sf0,'sfb=',sfb,'B0=',B0,'Uloop=',Uloop)
sf=saf_fact(sf0,sfb,run_cfg.r,a,Uloop)
R,Btotini,Btorini,Bpolini,Bpol1,Bradini,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitorini,dpsidr,dpsidfi,sf \
=Mag_field(run_cfg.r, run_cfg.thet, run_cfg.fi, B0, sf0, sfb, Uloop, run_cfg)
pperp2ini=run_cfg.pperp**2    
muini=pperp2ini/Btotini
p2ini=run_cfg.ppar**2+pperp2ini
psipolini=pi*B0*run_cfg.a**2/(sfb-sf0)*log((sf0+(sfb-sf0)*(run_cfg.r/run_cfg.a)**2)/sf0)
energyini=m01*ccc1**2*(sqrt(1+p2ini)-1)/1.6022e-12

logger.info('+++++++  start  +++++++++')

logger.info(f'rini= {run_cfg.r}, thetini={run_cfg.thet}, fiini={run_cfg.fi}, pparini= {run_cfg.ppar}, energyini= {energyini}')
#exit()

num_it=20
nrange=10000
delt=200000
logger.info(f"num_it= {num_it}, nrange= {nrange}")
logger.info(f"------------------------------------------------------------")
# Open the HDF5 file for writing (this will overwrite the old file)
start_time_full = time.time()
file_name ='results/full_trajectory.h5'
with pd.HDFStore(file_name, mode='w') as store:
    logger.info(f"Open the HDF5 file :  {file_name}")
    t_start = t_ini
    rini = run_cfg.r
    thetini = run_cfg.thet
    fiini = run_cfg.fi
    pparini = run_cfg.ppar
    
    for it in range(num_it):
        logger.info(f"Iteration {it}. Start")
        log_memory_usage()
        start_time = time.time()
        t0c=t_start
        sf0=spl_q0(t0c)
        sfb=spl_qa(t0c)
        Uloop=spl_U(t0c)
        B0=spl_B(t0c)
        logger.info(f't_start= {t0c}, sf0= {sf0}, sfb={sfb}, B0= {B0}, Uloop= {Uloop}')
        sf=saf_fact(sf0,sfb,rini,run_cfg.a,Uloop)
        logger.info(f'rini= {rini}, thetini= {thetini}, fiini= {fiini}, pparini= {pparini}')

        y0= [pparini, rini, thetini, fiini, pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini]
        t_end= t_start + delt  #t1UL
        logger.info(f'rini= {rini}, thetini= {thetini}, fiini= {fiini}, pparini= {pparini}, energyini= {energyini}')
        logger.info(f't_start(s)= {t_start*run_cfg.R0/ccc*tau_norm}, del_t_calculation(s)= {(t_end-t_start)*run_cfg.R0/ccc*tau_norm}, time(s)={t_end*run_cfg.R0/ccc*tau_norm}')
        #logger.info(f'solve_ivp: method= DOP853, t_eval={nrange}')
        logger.info(f'solve_ivp: method= DOP853, dense_output=True')
        sol= solve_ivp(fin_fun,
                    [t_start, t_end], 
                    y0, 
                    method='DOP853', 
                    dense_output=True, 
                    args=(run_cfg, muini),
                    rtol= 1e-7,
                    atol= 1e-10) 
        logger.info(f"Number of function evaluations {sol.nfev}")
        eval_time = time.time() - start_time
        logger.info(f"Number of function evaluations per sec {(sol.nfev/eval_time):0.2f}")

        t_steps = np.linspace(t_start, t_end, nrange)
        all_data = sol.sol(t_steps) # Получаем все данные разом!

        t_start= t_steps[-1]
        y_last = all_data[:, -1]
        pparini, rini, thetini, fiini, pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini = y_last

        thetini=thetini-int(thetini/(2*pi))*2*pi
        fiini=fiini-int(fiini/(2*pi))*2*pi

        df = pd.DataFrame(all_data.T, columns=['ppar','r','thet','fi','pperp2','Bpol','Btot','Brad','Btor','psipol','psitor','energy'])
        df['time'] =  t_steps

        logger.debug("\n" + df.head().to_string())
        logger.info(f"df size= {len(df)}")
        # Инкрементная запись в HDF5 
        store.append('trajectory', df, index=False)
        del df
        del sol
        del all_data
        gc.collect()
        logger.info(f"Iteration {it}. Execution time: {eval_time:0.2f} sec")
        logger.info(f"------------------------------------------------------------")

logger.info(f"Full execution time: {time.time() - start_time_full:0.2f} sec")        

#LSODA
#DOP853
# Сохраняем DataFrame в бинарный файл
#result_df.to_pickle('result.pkl') 