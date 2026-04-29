from line_profiler import profile
from loguru import logger
import numpy as np
import numba 
from numba import njit
numba.config.DISABLE_JIT = False # type: ignore
from hyp2f import create_fast_hyp, create_pade_hyp, fast_hyp2f1_specific, fast_hyp_part

from scipy.integrate import odeint 
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

#from numpy import pi, sin, cos, sqrt, log, tan, atan
from math import pi, sin, cos, sqrt, log, tan, atan

from config import RunConfig, RunParams
from field_EXL import *

from physical_constants import eqq, ccc, m0
from parameters import n

logger.info(f"Disable numba: {numba.config.DISABLE_JIT}") # type: ignore

@njit
def integrand(x):
    return x/sqrt(1-x**2)

@njit
def integrandn(x,n):
    return x**(n+1)/sqrt(1-x**2)

@njit
def integrandn1(x,n):
    if abs(x)>0.002:
        y=x**n*(1.-1./sqrt(1.-x**2))
    else:
        y=-x**(n+2)*0.5*(1.+3.*x**2/4)
    return y

@njit
def E0_field(r,thet,fi,R0,Uloop):
    E0tor=Uloop/(2*pi*R0)
    return E0tor

@njit    
def E_field(r,thet,fi,R0,E0tor):
    Etor=E0tor*R0/(R0+r*cos(thet))
    Erad=0.
    Epol=0.
    Etot=sqrt(Etor**2+Erad**2+Epol**2)
    if abs(Etot) >0.:
        etor=Etor/Etot
        erad=Erad/Etot
        epol=Epol/Etot
    else:
        etor=0.
        erad=0.
        epol=0.
    return Etot,Etor,etor,Erad,erad,Epol,epol

#def Bpol_f(r,thet,sf,B0,R0):
#    Bpol1=B0/(R0*sf)/(1+r/R0*cos(thet))
#    Bpol=Bpol1*r
#    return(Bpol,Bpol1)

@njit
def saf_fact(sf0,sfb,r,a,Uloop):
    sf=sf0+(sfb-sf0)*(r/a)**2
#    sf=sf*np.sign(Uloop)
    return sf

@njit
def fn(x,n):
    res1=0.
    if abs(x) >=5.e-2:
        for i in range(0,n):
            res1=res1+(-1)**i*x**(n-i)/(n-i)
        res1=res1+(-1)**n*log(abs(1.+x))
        res1=res1/(1+x)/x**(n+1)
    else:
        for i in range(0,10):
            res1=res1+(-1)**(n+i+2)*x**(+i)/(n+1+i)
        res1=res1*(-1)**n/(1+x)
    return res1 


#hyp_fast = create_fast_hyp(a = 0.5, b = (2.0 + n) / 2.0, c = (4.0 + n) / 2.0)
hyp_fast = create_pade_hyp(a = 0.5, b = (2.0 + n) / 2.0, c = (4.0 + n) / 2.0)
#@profile
@njit
def Mag_field(r, thet, fi, B0, sf0, sfb, Uloop, params :RunParams):
    #R0, a, delr, delfi, nfi, n, r, thet, fi, ppar, pperp
    R0, a, delr, delfi, nfi, n, _, _, _, _, _ = params
    x=r/R0
    R=R0+r*cos(thet)
    xpr=r*cos(thet)/R0
    Fnpr=fn(xpr,n+1)
    Btor=B0*R0/R*(1.+delfi*cos(nfi*fi)*(1.+delr*cos(thet))*(r/a)**n)
    Gpr=B0*delfi*nfi*sin(nfi*fi)*(1.+delr*cos(thet))
    Gpr1=B0*delfi*nfi**2*cos(nfi*fi)*(1.+delr*cos(thet))
    Gpr2=-B0*delfi*nfi*sin(nfi*fi)*delr*sin(thet)
    Gpr3=Gpr*x*(r/a)**n
    Gpr31=Gpr*x*(r/a)**(n-1)/a
    Fpr=(r/a)**n*Fnpr
    #if r<=0:
    #    print('x=',x,'R=',R,'cos(thet)=',cos(thet),'xpr=',xpr)
    if abs(xpr)>=0.01:  
        dFndthet=(Fnpr*((n+2)/xpr+1./(1.+xpr))-1./xpr/(1.+xpr)**2)*x*sin(thet)
    else:
        apr=0.
        bpr=0.
        for i in range(0,10):
            apr=+apr+(-1)**(n+1+i)*xpr**i/(n+2.+i)
        for i in range(1,10):
            bpr=bpr+(-1)**(n+1+i)*i*xpr**(i-1)/(n+2.+i)
        dFndthet=-((-1)**n/(1+xpr)**2*apr+(-1)**(n+1)/(1+xpr)*bpr)*x*sin(thet)    

    Brad1=Gpr*(1./R0)*Fpr
    Brad=r*Brad1

    psi0=2*pi*B0*R0**2
    psi0n=psi0*(R0/a)**n

    if x>0.02:
        res=1.-sqrt(1.-x**2)
    else:
        res=x**2/2.*(1.+x**2/4.*(1.+x**2/2.*(1.+5.*x**2/16.)))

    A1=psi0*res

    #resn=x**(n+2)*fast_hyp_part(x, n)/(2+n)
    resn=x**(n+2)*hyp_fast(x)/(2+n)

    An=psi0n*resn

    resn1=-x**(1+n)*(-1.+fast_hyp2f1_specific(x,n))/(1+n)
    #resn1=-x**(1+n)*(-1.+hyp_fast(x))/(1+n)
    
    An1=psi0n*resn1

    psitor=A1+(An+delr*An1)*delfi*cos(nfi*fi)

    A1=psitor
    rpsi=(R0/abs(psi0))*sqrt((2*psi0-A1)*A1)
    sf=saf_fact(sf0,sfb,rpsi,a,Uloop)
    #sf1=saf_fact(sf0,sfb,r,a,Uloop)

    dpsidA1=1.
    dA1dr=(psi0/R0)*x/sqrt(1-x**2)
    dpsidAn=delfi*cos(nfi*fi)
    dAndr=psi0n/R0*integrandn(x,n)     #*(n+1+x**2/(1.-x**2))
    dpsidAn1=delr*delfi*cos(nfi*fi)

    dAn1dr=psi0n*integrandn1(x,n)
    dpsidr=dpsidA1*dA1dr+dpsidAn*dAndr+dpsidAn1*dAn1dr
    dsfdpsi=2.*(sfb-sf0)*(psi0-psitor)*(R0/a)**2/psi0**2*np.sign(B0)
    dsfdr=dsfdpsi*dpsidr
    Bpol=dpsidr/sf/(R*2.*pi)*np.sign(B0)

    Bpol1=Bpol/r
    dpsidAndfi=-nfi*delfi*sin(nfi*fi)
    dpsidAn1dfi=-nfi*delr*delfi*sin(nfi*fi)
    dpsidrdfi=(dpsidAndfi*dAndr+dpsidAn1dfi*dAn1dr)
    dpsidfi=-(An+delr*An1)*nfi*delfi*sin(nfi*fi)
    dsfdfi=dsfdpsi*dpsidfi

    Btot=sqrt(Btor**2+Bpol**2+Brad**2)
    brad=Brad/Btot
    bpol=Bpol/Btot
    bpol1=Bpol1/Btot
    btor=Btor/Btot

    dBtordr=-Btor*cos(thet)/R+B0*R0/R*delfi*cos(nfi*fi)*(1+delr*cos(thet))*(r/a)**(n-1)*n/a

    dA1drdr=psi0/(R0**2*(sqrt(1-x**2))**3)
    dAndrdr=psi0n*x**n*(n+1-n*x**2)/(R0**2*(sqrt(1-x**2))**3)
    dAn1drdr=psi0n/R0*(x**(n-1)*(n*(1-1/sqrt(1-x**2))-x**2/(sqrt(1.-x**2))**3))
    dpsidrdr=dpsidA1*dA1drdr+dpsidAn*dAndrdr+dpsidAn1*dAn1drdr


    dBpoldr=(dpsidrdr-dpsidr*dsfdr/sf*np.sign(B0)-dpsidr*cos(thet)/R)/(2*pi*R*sf)*np.sign(B0) 

    dBtordfi=-B0/(1+xpr)*delfi*nfi*sin(nfi*fi)*(1.+delr*cos(thet))*(r/a)**n
    dBraddr=(-Brad1*(R0+2.*r*cos(thet))-dBtordfi)/R

    dBpoldfi=(dpsidrdfi/(2*pi*R*sf)-dpsidr*dsfdfi/(sf**2*2.*pi*R))*np.sign(B0)

    dBraddfi=Gpr1*(r/R0)*Fpr
    dBpoldthet1=Bpol*sin(thet)/R
    dBpoldthet=dBpoldthet1*r

    dBtordthet1=Btor*sin(thet)/R-B0*delfi*cos(nfi*fi)*delr*sin(thet)/(1+xpr)*(r/a)**(n-1)/a
    dBtordthet=dBtordthet1*r

    dBraddthet1=Gpr2*(1/R0)*Fpr+Gpr31*dFndthet
    dBraddthet=dBraddthet1*r

    return R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitor,dpsidr,dpsidfi,sf

@njit
def rot_b(r,thet,fi,R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1):
    br=brad
    bpol=bpol
    bpol1=Bpol1/Btot
    bfi=btor

    dbrdthet1=(dBraddthet1-brad*(brad*dBraddthet1+bpol*dBpoldthet1+btor*dBtordthet1))/Btot
    dbrdthet=(dBraddthet-brad*(brad*dBraddthet+bpol*dBpoldthet+btor*dBtordthet))/Btot
    dbrdfi=(dBraddfi-brad*(brad*dBraddfi+bpol*dBpoldfi+btor*dBtordfi))/Btot

    dbpoldr=(dBpoldr-bpol*(brad*dBraddr+bpol*dBpoldr+btor*dBtordr))/Btot
    dbpoldfi=(dBpoldfi-bpol*(brad*dBraddfi+bpol*dBpoldfi+btor*dBtordfi))/Btot

    dbfidr=(dBtordr-btor*(brad*dBraddr+bpol*dBpoldr+btor*dBtordr))/Btot
    dbfidthet1=(dBtordthet1-btor*(brad*dBraddthet1+bpol*dBpoldthet1+btor*dBtordthet1))/Btot
    dbfidthet=(dBtordthet-btor*(brad*dBraddthet+bpol*dBpoldthet+btor*dBtordthet))/Btot
#    brtr=(dbpoldfi+bfi*sin(thet))/R-dbfidthet/r

    rtbr=(dbpoldfi+bfi*sin(thet))/R-dbfidthet1
    rtbpol=(bfi*cos(thet)-dbrdfi)/R+dbfidr
    rtbfi=(dbrdthet1-bpol1)-dbpoldr

    #   this is correct
    brtr=(bpol*rtbfi-bfi*rtbpol)
    brtt=(bfi*rtbr-br*rtbfi)
    brtfi=(br*rtbpol-bpol*rtbr)
#########################################
    
    gbr=(Brad*dBraddr+Bpol*dBpoldr+Btor*dBtordr)/Btot
    gbt=(Brad*dBraddthet1+Bpol*dBpoldthet1+Btor*dBtordthet1)/Btot
    gbfi=(Brad*dBraddfi+Bpol*dBpoldfi+Btor*dBtordfi)/Btot/R

    glbr=gbr/Btot
    glbt=gbt/Btot
    glbfi=gbfi/Btot

    bgrr=bpol*glbfi-btor*glbt
    bgrt=btor*glbr-br*glbfi
#    bgrt1=btor*gbr1-br*gbfi1
    bgrfi=br*glbt-bpol*glbr

    bbrtr=bpol*brtfi-bfi*brtt
    bbrtt=bfi*brtr-br*brtfi
#bbrtt1=bfi*brtr-br*brtfi
    bbrtfi=br*brtt-bpol*brtr
    return rtbr,rtbpol,rtbfi,brtr,brtt,brtfi,gbr,gbt,gbfi,bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi

@njit
def eq_mot(t, R0,pperp,ppar,r,thet,fi,R,Uloop,brtr,brtt,brtfi,gbr,gbt,gbfi, \
    bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi,brad,btor,bpol,muini,Btot): #,dBpoldr,dBpoldthet,dBpoldfi, \
    # dBraddr,dBraddthet,dBraddfi,dBtordr,dBtordthet,dBtordfi,Bpol,Brad,Btor,psitor,dpsidr,dpsidfi,sf):
    
    E0tor=E0_field(r,thet,fi,R0,Uloop)
    Etot,Etor,etor,Erad,erad,Epol,epol=E_field(r,thet,fi,R0,E0tor)
    ptot2=pperp**2+ppar**2
    gam=sqrt(1.+ptot2)

    dppardt=eqq*R0/(m0*ccc**2)*(Erad*brad+Epol*bpol+Etor*btor)-R0*pperp**2/(2*Btot*gam)* \
    (gbr*brad+gbt*bpol+gbfi*btor)

    omce=eqq*Btot/(m0*ccc)
    M1=ppar*R0/gam
    M2=0.5*R0/(eqq/(m0*ccc)*gam)*muini   #correct
    M3=R0/(omce*gam)*ppar**2
 #   M4=R0/(ccc*Btot)
    M4=0
    dRdtr=M1*brad+M2*bgrr+M3*bbrtr+M4*(Epol*btor-Etor*bpol)
    dRdtt=M1*bpol+M2*bgrt+M3*bbrtt+M4*(Etor*brad-Erad*btor)
    dRdtfi=M1*btor+M2*bgrfi+M3*bbrtfi+M4*(Erad*bpol-Epol*brad)
    y1=dppardt
    y2=dRdtr
    y3=dRdtt/r
    y4=dRdtfi/R
    # dpperp2dt=muini*(gbr*y2+gbt*y3*r+gbfi*y4*R)

    # y5=dpperp2dt
    # dBpoldt=dBpoldr*y2+dBpoldthet*y3+dBpoldfi*y4
    # y6=dBpoldt
    # dBtotdt=gbr*y2+gbt*y3*r+gbfi*y4*R
    # y7=dBtotdt
    # dBraddt=dBraddr*y2+dBraddthet*y3+dBraddfi*y4
    # y8=dBraddt
    # dBtordt=dBtordr*y2+dBtordthet*y3+dBtordfi*y4
    # y9=dBtordt
    # y10=R*R0*Etor/ccc+(y4*dpsidfi+y2*dpsidr)/(sf*2.*pi)*np.sign(btor)
    # y11=(y4*dpsidfi+y2*dpsidr)
    # y12=-Etor*dRdtfi

    return [y1, y2, y3, y4] #,y5,y6,y7,y8,y9,y10,y11,y12] 

# --- Функция события: пересечение границы ---
def hit_wall(t, y):
    #R, phi, Z, v_p = y
    ppar, r, thet, fi = y
    # 
    return r - 0.6
# Настройка события: остановить интеграцию при достижении 0
hit_wall.terminal = True 
hit_wall.direction = 1  # Срабатывает только при вылете наружу

def guiding_center_dynamics(t, y, params:RunParams, muini):
    #a,R0,delr,delfi,nfi,n,pparini,pperpini
    ppar, r, thet, fi = y
    
    sf0=spl_q0(t)
    sfb=spl_qa(t)
#    sfb=Splines.spl_qa(t)
    Uloop=spl_U(t)
    B0=spl_B(t)
    sf=saf_fact(sf0,sfb,r,params.a,Uloop)

    R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitor,dpsidr,dpsidfi,sf \
    =Mag_field(r, thet, fi, B0, sf0, sfb, Uloop, params)

    rtbr,rtbpol,rtbfi,brtr,brtt,brtfi,gbr,gbt,gbfi,bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi  \
    =rot_b(r,thet,fi,R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,   \
    dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1)

    pperp2=muini*Btot
    pperp=sqrt(pperp2)

    dydt=eq_mot(t, params.R0,pperp,ppar,r,thet,fi,R,Uloop,brtr,brtt,brtfi,gbr,gbt,gbfi, \
    bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi,brad,btor,bpol,muini,Btot) #,dBpoldr,dBpoldthet,dBpoldfi,  \
    #dBraddr,dBraddthet,dBraddfi,dBtordr,dBtordthet,dBtordfi,Bpol,Brad,Btor,psitor,dpsidr,dpsidfi,sf)

    return dydt 


