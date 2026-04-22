import tomllib
from typing import NamedTuple
from logger_config import logger

class RunConfig(NamedTuple):
    R0: float
    a: float
    delr: float
    delfi: float
    nfi: int
 
def load_configs(discharge_path):
    with open(discharge_path, "rb") as f:
        cfg = tomllib.load(f)
        
    logger.info(f"Tokamak: {cfg['tokamak']['name']}")

    return RunConfig(
        R0= cfg['tokamak']['R0'],
        a= cfg['tokamak']['a'],
        delr= cfg['tokamak']['delr'],
        delfi= cfg['tokamak']['delfi'],
        nfi=cfg['tokamak']['nfi']
    )

def log_config(cfg):
    logger.info(f"R0 = {cfg.R0} a = {cfg.a}")
    logger.info(f"delr = {cfg.delr} delr = {cfg.delr}")
    logger.info(f"nfi = {cfg.nfi}")
