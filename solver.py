import gc
import pandas as pd
import config 
from logger_config import log_memory_usage, logger
import time
from scipy.integrate import odeint,solve_ivp  

from physical_constants import *
import parameters
        

run_cfg = config.load_configs('discharges/base_shot.toml')

logger.info(f"Tokamak: {run_cfg.tokamak_name} Shot number: {run_cfg.shot_number}")
logger.info(config.param_string(run_cfg.params))

# eval const
params = run_cfg.params
ccc_R0 = ccc/params.R0
parameters.ccc_R0 = ccc_R0
parameters.a = params.a
parameters.R0 = params.R0
parameters.n = params.n
from eqations import *

t_ini = run_cfg.time_start*ccc_R0/tau_norm

t0c=t_ini
sf0=spl_q0(t0c)
sfb=spl_qa(t0c)
Uloop=spl_U(t0c)
B0=spl_B(t0c)
#print('t_ini=',t0c,'sf0=',sf0,'sfb=',sfb,'B0=',B0,'Uloop=',Uloop)
sf=saf_fact(sf0,sfb,params.r,a,Uloop)
R,Btotini,Btorini,Bpolini,Bpol1,Bradini,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitorini,dpsidr,dpsidfi,sf \
=Mag_field(params.r, params.thet, params.fi, B0, sf0, sfb, Uloop, params)
pperp2ini=params.pperp**2    
muini=pperp2ini/Btotini
p2ini=params.ppar**2+pperp2ini
psipolini=pi*B0*params.a**2/(sfb-sf0)*log((sf0+(sfb-sf0)*(params.r/params.a)**2)/sf0)
energyini=m01*ccc1**2*(sqrt(1+p2ini)-1)/1.6022e-12

logger.info('+++++++  start  +++++++++')

logger.info(f'rini= {params.r}, thetini={params.thet}, fiini={params.fi}, pparini= {params.ppar}, energyini= {energyini}')

logger.info(f"------------------------------------------------------------")
# Open the HDF5 file for writing (this will overwrite the old file)
calculation_start_time = time.time()
file_name ='results/full_trajectory.h5'
with pd.HDFStore(file_name, mode='w') as store:
    logger.info(f"Open the HDF5 file :  {file_name}")
    tau_start = t_ini
    rini = params.r
    thetini = params.thet
    fiini = params.fi
    pparini = params.ppar
    logger.info(f"num_it= {run_cfg.num_it}, nrange= {run_cfg.nrange}")
    for it in range(run_cfg.num_it):
        logger.info(f"Iteration {it}. Start")
        log_memory_usage()
        iteration_start_time = time.time()
        t0c=tau_start
        sf0=spl_q0(t0c)
        sfb=spl_qa(t0c)
        Uloop=spl_U(t0c)
        B0=spl_B(t0c)
        logger.info(f'tau_start= {t0c}, sf0= {sf0}, sfb={sfb}, B0= {B0}, Uloop= {Uloop}')
        sf=saf_fact(sf0,sfb,rini,params.a,Uloop)
        
        y0= [pparini, rini, thetini, fiini] #, pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini]
        tau_end= tau_start + run_cfg.delta_tau  #t1UL

        logger.info(f'rini= {rini}, thetini= {thetini}, fiini= {fiini}, pparini= {pparini}, energyini= {energyini}')
        logger.info(f't_start(s)= {tau_start*params.R0/ccc*tau_norm}, del_t_calculation(s)= {(tau_end-tau_start)*params.R0/ccc*tau_norm}, time(s)={tau_end*params.R0/ccc*tau_norm}')
        #logger.info(f'solve_ivp: method= DOP853, t_eval={nrange}')
        logger.info(f'solve_ivp: method= DOP853, dense_output=True')
        sol= solve_ivp(fin_fun,
                    [tau_start, tau_end], 
                    y0, 
                    method='DOP853', 
                    dense_output=True, 
                    args=(params, muini),
                    rtol= 1e-7,
                    atol= 1e-10) 
        logger.info(f"Number of function evaluations {sol.nfev}")
        iteration_time = time.time() - iteration_start_time
        logger.info(f"Number of function evaluations per sec {(sol.nfev/iteration_time):0.2f}")

        t_steps = np.linspace(tau_start, tau_end, run_cfg.nrange)
        all_data = sol.sol(t_steps) # Получаем все данные разом!

        tau_start= t_steps[-1]
        y_last = all_data[:, -1]
        #pparini, rini, thetini, fiini , pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini = y_last
        pparini, rini, thetini, fiini = y_last

        theta_revolutions = thetini/(2*pi)
        fi_revolutions = fiini/(2*pi)
        logger.info(f'theta_revolutions= {theta_revolutions:0.2f}, fi_revolutions= {fi_revolutions:0.2f}')
        thetini=thetini-int(theta_revolutions)*2*pi
        fiini=fiini-int(fi_revolutions)*2*pi
        
        df = pd.DataFrame(all_data.T, columns=['ppar','r','thet','fi'])
        df['tau'] =  t_steps

        logger.debug("\n" + df.head().to_string())
        logger.info(f"df size= {len(df)}")
        # Инкрементная запись в HDF5 
        store.append('trajectory', df, index=False)
        del df
        del sol
        del all_data
        gc.collect()
        logger.info(f"Iteration {it}. calculation time: {iteration_time:0.2f} sec")
        logger.info(f"------------------------------------------------------------")

logger.info(f"Full calculationtime: {time.time() - calculation_start_time:0.2f} sec")        

#LSODA
#DOP853
# Сохраняем DataFrame в бинарный файл
#result_df.to_pickle('result.pkl') 