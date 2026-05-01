import tomllib
from typing import NamedTuple


class RunParams(NamedTuple):
    R0: float
    a: float
    delr: float
    delfi: float
    nfi: int
    n: int
    r: float
    theta: float
    phi: float
    ppar: float
    pperp: float

class RunConfig(NamedTuple):
    tokamak_name: str
    shot_number: int
    time_start: float # [sec]
    num_it: int
    nrange: int
    delta_tau: float
    params: RunParams




def load_configs(discharge_path):
    with open(discharge_path, "rb") as f:
        cfg = tomllib.load(f)
    params = RunParams(
        R0=    cfg['tokamak']['R0'],
        a=     cfg['tokamak']['a'],
        delr=  cfg['discharge']['perturbations']['delr'],
        delfi= cfg['discharge']['perturbations']['delfi'],
        nfi=   cfg['discharge']['perturbations']['nfi'],
        n=     cfg['discharge']['perturbations']['n'],
        r=     cfg['initial_conditions']['r'],
        theta=  cfg['initial_conditions']['theta'],
        phi=    cfg['initial_conditions']['phi'],
        ppar=  cfg['initial_conditions']['ppar'],
        pperp= cfg['initial_conditions']['pperp'],
    )
    return RunConfig(
        tokamak_name = cfg['tokamak']['name'],
        shot_number = cfg['discharge']['main']['shot_number'],
        time_start=  cfg['initial_conditions']['time_start'],
        num_it=      cfg['initial_conditions']['num_it'],
        nrange=      cfg['initial_conditions']['nrange'],
        delta_tau=   cfg['initial_conditions']['delta_tau'],
        params= params
    )

def param_string(p:RunParams):
    info = f"R0 = {p.R0}, a = {p.a}, "
    info += f"delr = {p.delr}, delr = {p.delr}, "
    info += f"nfi = {p.nfi}"
    return info

