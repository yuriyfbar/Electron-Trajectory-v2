# from math import tan, atan
import numpy as np
import scipy.special as sc
from scipy.integrate import odeint 
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
import pandas as pd
#from mgn_best_DOP853 import saf_fact, Mag_field, fin_fun, eq_mot, rot_b, eq_mot_1

from numpy import *  #pi, sin, cos, sqrt, log, tan, atan

#from field import *
#from field_FT2 import *
from field_EXL_50U_13976 import *


def integrand(x):
    return x/sqrt(1-x**2)
def integrandn(x,n):
    return(x**(n+1)/sqrt(1-x**2))
def integrandn1(x,n):
    if abs(x)>0.002:
        y=x**n*(1.-1./sqrt(1.-x**2))
    else:
        y=-x**(n+2)*0.5*(1.+3.*x**2/4)
    return(y)

def E0_field(r,thet,fi,R0,Uloop):
    E0tor=Uloop/(2*pi*R0)
    return(E0tor)
    
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
    return(Etot,Etor,etor,Erad,erad,Epol,epol)

#def Bpol_f(r,thet,sf,B0,R0):
#    Bpol1=B0/(R0*sf)/(1+r/R0*cos(thet))
#    Bpol=Bpol1*r
#    return(Bpol,Bpol1)

def saf_fact(sf0,sfb,r,a,Uloop):
    sf=sf0+(sfb-sf0)*(r/a)**2
#    sf=sf*np.sign(Uloop)
    return(sf)

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
    return(res1)

def Mag_field(r,thet,fi,R0,a,B0,delfi,nfi,delr,n,sf0,sfb,Uloop):
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
#    print(xpr,dFndthet)
#    sys.exit()
    Brad1=Gpr*(1./R0)*Fpr
    Brad=r*Brad1
#    print('r,Brad',r,Brad,'Gpr,Fpr',Gpr,Fpr)
#######################################3333
#    poloidal field calculation
    psi0=2*pi*B0*R0**2
    psi0n=psi0*(R0/a)**n
#    psi0n1=psi0/R0/((n+1)*a**n)
#    A=psi0*(1-sqrt(1-x**2))
#    if(x**2>1):
#        print('x=',x,'r=',r)
#        exit()
    if x>0.02:
        res=1.-sqrt(1.-x**2)
    else:
        res=x**2/2.*(1.+x**2/4.*(1.+x**2/2.*(1.+5.*x**2/16.)))
#    print(x,res,res1)
#    sys.exit()
    A1=psi0*res
#    print(res)
#    sys.exit()
#    A1=psi0*res[0]
#    print(A,A1,A1/A,psi0)
#    print('A1',A1)
    resn=x**(n+2)*sc.hyp2f1(0.5,(2.+n)/2,(4.+n)/2,x**2)/(2+n)
#    print('resn',resn)
    An=psi0n*resn
#    print('resn',resn)
#    An=psi0n*resn[0]
#    print('An',An)
#    resn1=quad(integrandn1,0.,r/R0,args=(n))
#    An1=psi0n*resn1[0]
    resn1=-x**(1+n)*(-1.+sc.hyp2f1(0.5,(1.+n)/2,(3.+n)/2,x**2))/(1+n)
    An1=psi0n*resn1

#    print('An1',An1)
    psitor=A1+(An+delr*An1)*delfi*cos(nfi*fi)
#    psitor1=pi*r**2*B0

#    sys.exit()
    A1=psitor
    rpsi=(R0/abs(psi0))*sqrt((2*psi0-A1)*A1)
    sf=saf_fact(sf0,sfb,rpsi,a,Uloop)
    sf1=saf_fact(sf0,sfb,r,a,Uloop)

#    Bpol,Bpol1=Bpol_f(r,thet,sf,B0,R0)
#    print('Bpol',Bpol)
    
    dpsidA1=1.
    dA1dr=(psi0/R0)*x/sqrt(1-x**2)
    dpsidAn=delfi*cos(nfi*fi)
    dAndr=psi0n/R0*integrandn(x,n)     #*(n+1+x**2/(1.-x**2))
    dpsidAn1=delr*delfi*cos(nfi*fi)
#    print('psi0n1',psi0n1,n,r**(n-1))
    dAn1dr=psi0n*integrandn1(x,n)
    dpsidr=dpsidA1*dA1dr+dpsidAn*dAndr+dpsidAn1*dAn1dr
    dsfdpsi=2.*(sfb-sf0)*(psi0-psitor)*(R0/a)**2/psi0**2*np.sign(B0)
    dsfdr=dsfdpsi*dpsidr
    Bpol=dpsidr/sf/(R*2.*pi)*np.sign(B0)
#    sys.exit()
    Bpol1=Bpol/r
    dpsidAndfi=-nfi*delfi*sin(nfi*fi)
    dpsidAn1dfi=-nfi*delr*delfi*sin(nfi*fi)
    dpsidrdfi=(dpsidAndfi*dAndr+dpsidAn1dfi*dAn1dr)
    dpsidfi=-(An+delr*An1)*nfi*delfi*sin(nfi*fi)
    dsfdfi=dsfdpsi*dpsidfi
#    print('dpsidA1*dA1dr=',dpsidA1*dA1dr,'dpsidAn*dAndr=',dpsidAn*dAndr,'dpsidAn1*dAn1dr=',dpsidAn1*dAn1dr)
#    print('dpsidr=',dpsidr,'dsfdpsi*dsfdr=', dsfdpsi*dsfdr)
#    print(dpsidA1*dA1dr,dpsidAn*dAndr,dpsidAn1*dAn1dr,dpsidr,dsfdpsi*dsfdr)
    
#    dBtordfinp
    ########################33
    Btot=sqrt(Btor**2+Bpol**2+Brad**2)
    brad=Brad/Btot
    bpol=Bpol/Btot
    bpol1=Bpol1/Btot
    btor=Btor/Btot
#    print('brad,bpol,btor=',brad,bpol,btor)
#    print('Brad,Bpol,Btor=',Brad,Bpol,Btor)
#    sys.exit()
    
    dBtordr=-Btor*cos(thet)/R+B0*R0/R*delfi*cos(nfi*fi)*(1+delr*cos(thet))*(r/a)**(n-1)*n/a
#######################################################
#    dBpoldr=(B0/R)/sf*(1.-xpr/(1.+xpr)-r*dsfdr/sf)
    dA1drdr=psi0/(R0**2*(sqrt(1-x**2))**3)
    dAndrdr=psi0n*x**n*(n+1-n*x**2)/(R0**2*(sqrt(1-x**2))**3)
    dAn1drdr=psi0n/R0*(x**(n-1)*(n*(1-1/sqrt(1-x**2))-x**2/(sqrt(1.-x**2))**3))
    dpsidrdr=dpsidA1*dA1drdr+dpsidAn*dAndrdr+dpsidAn1*dAn1drdr

#    dBpoldr=(dpsidrdr-dpsidr*dsfdr/abs(sf)-dpsidr*cos(thet)/R)/(2*pi*R*sf)*np.sign(B0)   #original
#    dBpoldr=(dpsidrdr-dpsidr*dsfdr/sf-dpsidr*cos(thet)/R)/(2*pi*R*sf)*np.sign(B0)
#    dBpoldr=(dpsidrdr*np.sign(B0)-dpsidr*dsfdr/abs(sf)-dpsidr*cos(thet)/R)/(2*pi*R*sf)   #correct
    dBpoldr=(dpsidrdr-dpsidr*dsfdr/sf*np.sign(B0)-dpsidr*cos(thet)/R)/(2*pi*R*sf)*np.sign(B0) 
#
#    print('dpsidr*cos(thet)/R=',dpsidr*cos(thet)/R)
#    print('dBpoldr=(dpsidrdr-dpsidr*dsfdr/sf-dpsidr*cos(thet)/R)/(2*pi*R*sf)*np.sign(B0)=',  \
#    (dpsidrdr-dpsidr*dsfdr/sf-dpsidr*cos(thet)/R)/(2*pi*R*sf)*np.sign(B0))
#######################################################
    dBtordfi=-B0/(1+xpr)*delfi*nfi*sin(nfi*fi)*(1.+delr*cos(thet))*(r/a)**n
    dBraddr=(-Brad1*(R0+2.*r*cos(thet))-dBtordfi)/R
#    dBpoldfi=-(Bpol/sf)*((sfb-sf0)/a**2)*(R0**2/psi0)*((An+delr*An1)*delfi*nfi*sin(nfi*fi))
#    dBpoldfi=2.*(Bpol/sf)*((sfb-sf0)/a**2)*(R0**2/psi0**2*(psi0-psitor))*((An+delr*An1)*delfi*nfi*sin(nfi*fi))
#    dBpoldfi=2*B0*x/(1+xpr)/sf**2*(sfb-sf0)*(R0/a/psi0)**2*(psi0-psitor)*(delfi*An+delr*delfi*An1)*sin(nfi*fi)
#    dBpoldfi=2*B0*x/(1+xpr)/sf**2*(sfb-sf0)*(R0/a/psi0)**2*(psi0-psitor)*delfi*nfi*sin(nfi*fi)*An
##################################################################
#    dBpoldfi=(dpsidrdfi/(2*pi*R*sf)-dpsidr*dsfdfi/(sf**2*2.*pi*R))    #original
    dBpoldfi=(dpsidrdfi/(2*pi*R*sf)-dpsidr*dsfdfi/(sf**2*2.*pi*R))*np.sign(B0)
#############################################################
#    print('dpsidrdfi/(2*pi*R*sf)=',dpsidrdfi/(2*pi*R*sf),'dpsidr*dsfdfi/(sf**2*2.*pi*R)=',dpsidr*dsfdfi/(sf**2*2.*pi*R))
#    dBpoldfi1=abs(dpsidrdfi/(2*pi*R*sf)-dpsidr*dsfdfi/(sf**2*2.*pi*R))*np.sign(Uloop*B0)
#    print('dBpoldfi1=',dBpoldfi1,'dBpoldfi=',dBpoldfi)
#    print('dpsidrdfi/(2*pi*R*sf)=',dpsidrdfi/(2*pi*R*sf),'dpsidr*dsfdfi/(sf**2*2.*pi*R)=',dpsidr*dsfdfi/(sf**2*2.*pi*R))
#    sys.exit()
    dBraddfi=Gpr1*(r/R0)*Fpr
    dBpoldthet1=Bpol*sin(thet)/R
    dBpoldthet=dBpoldthet1*r
#    print('dBpoldr1=',dBpoldr1,'dBpoldr=',dBpoldr,'dBpoldthet=',dBpoldthet)

#    sys.exit()
#    dBtordthet1=Btor*r*sin(thet)/R0/(1.+xpr)-B0*delfi*cos(nfi*fi)*delr*sin(thet)/(1+xpr)*(r/a)**(n-1)/a
#    dBtordthet1=Btor*sin(thet)/R0/(1.+xpr)-B0*delfi*cos(nfi*fi)*delr*sin(thet)/(1+xpr)*(r/a)**(n-1)/a
    dBtordthet1=Btor*sin(thet)/R-B0*delfi*cos(nfi*fi)*delr*sin(thet)/(1+xpr)*(r/a)**(n-1)/a
    dBtordthet=dBtordthet1*r
#    dFndthet=[
    dBraddthet1=Gpr2*(1/R0)*Fpr+Gpr31*dFndthet
    dBraddthet=dBraddthet1*r
    
   
#    dBtotdr=Brad*dBraddr+Bpol*dBpoldr+Btor*dBtordr
#    dBtotdthet=Brad*dBraddthet+Bpol*dBpoldthet+Btor*dBtordthet
#    dBtotdfi=Brad*dBraddfi+Bpol*dBpoldfi+Btor*dBtordfi
#    print('dBraddfi=',dBraddfi,'dBpoldfi=',dBpoldfi,'dBtordfi=',dBtordfi,'dBtotdfi=',dBtotdfi)
#    print('Brad=',Brad,' Bpol=',Bpol,' Btor=',Btor)
#    print('sf=',sf)
#    print('Brad=',Brad,'dBraddr=',dBraddr,'dBraddthet=',dBraddthet,'dBraddfi=',dBraddfi)

#    print('psi0=',psi0,'psi0n=',psi0n)    
#    print('rpsi',rpsi)
#    print('sf=',sf,'sf1=',sf1)    
#    print('psitor=',psitor)
#    print('dpsidr=',dpsidr,'dsfdpsi=',dsfdpsi,'dsfdr=dsfdpsi*dpsidr=',dsfdr)
#    print('Bpol=dpsidr/sf/(R*2.*pi)*np.sign(B0)=',dpsidr/sf/(R*2.*pi)*np.sign(B0))
#    print('dBpoldr=',dBpoldr,'dpsidrdr=',dpsidrdr,'dpsidr*dsfdr/sf=',dpsidr*dsfdr/sf)
#    print('dBpoldr=',dBpoldr,'dBpoldthet=',dBpoldthet,'dBtordthet1=',dBtordthet1,'dBtordthet=',dBtordthet)
#    sys.exit()

    return(R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitor,dpsidr,dpsidfi,sf)
############################################################

##############
#   dbrdthet=(dBraddthet-Brad*(Brad*dBraddthet+Bpol*dBpoldthet+Btor*dBtordthet)/Btot**2)/Btot
#   dbpoldr=(dBpoldr-Bpol*(Brad*dBraddr+Bpol*dBpoldr+Btor*dBtordr)/Btot**2)/Btot
#   rtbfi=(dbrdthet-bpol)/r-dbpoldr                                         
#   dbfidthet=(dBtordthet-Btor*(Brad*dBraddthet+Bpol*dBpoldthet+Btor*dBtordthet)/Btot**2)/Btot  

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
###############################################
    # this is wrong
#    brtr=-(bpol*rtbfi-bfi*rtbpol)
#    brtt=-(bfi*rtbr-br*rtbfi)
#    brtfi=-(br*rtbpol-bpol*rtbr)
###########################################
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
    return(rtbr,rtbpol,rtbfi,brtr,brtt,brtfi,gbr,gbt,gbfi,bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi)

def eq_mot(t,eqq,ccc,m0,R0,pperp,ppar,r,thet,fi,R,Uloop,brtr,brtt,brtfi,gbr,gbt,gbfi, \
    bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi,brad,btor,bpol,muini,Btot,dBpoldr,dBpoldthet,dBpoldfi, \
    dBraddr,dBraddthet,dBraddfi,dBtordr,dBtordthet,dBtordfi,Bpol,Brad,Btor,psitor,dpsidr,dpsidfi,sf):
    
    E0tor=E0_field(r,thet,fi,R0,Uloop)
    Etot,Etor,etor,Erad,erad,Epol,epol=E_field(r,thet,fi,R0,E0tor)
    ptot2=pperp**2+ppar**2
    gam=sqrt(1.+ptot2)


    dppardt=eqq*R0/(m0*ccc**2)*(Erad*brad+Epol*bpol+Etor*btor)-R0*pperp**2/(2*Btot*gam)* \
    (gbr*brad+gbt*bpol+gbfi*btor)
#    print('dppardt=',dppardt,'eqq*R0/(m0*ccc**2)*(Erad*brad+Epol*bpol+Etor*btor)=',eqq*R0/(m0*ccc**2)*(Erad*brad+Epol*bpol+Etor*btor))
#    print('eqq=',eqq,'(Erad*brad+Epol*bpol+Etor*btor)=',(Erad*brad+Epol*bpol+Etor*btor))
#    print('Etor*btor=',Etor*btor,'Etor=',Etor)
#    sys.exit()
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
    dpperp2dt=muini*(gbr*y2+gbt*y3*r+gbfi*y4*R)
#    print('dBtotdr=',dBtotdr,'dBtotdthet=',dBtotdthet,'dBtotdfi=',dBtotdfi)
#    sys.exit()
#    dpperp2dt=muini/Btot*(dBtotdr*y2+dBtotdthet*y3)
#    dpperp2dt=muini/Btot*(dBtotdr*y2+dBtotdfi*y4)
#    dpperp2dt=muini/Btot*(dBtotdthet*y3+dBtotdfi*y4)
#    dpperp2dt=muini/Btot*(dBtotdfi*y4)
    y5=dpperp2dt
#    dBpoldt=dBpoldr*y2           #+dBpoldthet*y3+dBpoldfi*y4
##############################################################
#    dBpoldt=dBpoldr*y2+dBpoldthet*y3+dBpoldfi*y4   #original
    dBpoldt=dBpoldr*y2+dBpoldthet*y3+dBpoldfi*y4
#############################################3333
    y6=dBpoldt
    dBtotdt=gbr*y2+gbt*y3*r+gbfi*y4*R
    y7=dBtotdt
    dBraddt=dBraddr*y2+dBraddthet*y3+dBraddfi*y4
#    print('Brad=',Brad,'dBraddr=',dBraddr,'dBraddthet=',dBraddthet,'dBraddfi=',dBraddfi)
#    sys.exit()
    y8=dBraddt
    dBtordt=dBtordr*y2+dBtordthet*y3+dBtordfi*y4
    y9=dBtordt
    y10=R*R0*Etor/ccc+(y4*dpsidfi+y2*dpsidr)/(sf*2.*pi)*np.sign(btor)
#    y10=R*R0*Etor/ccc-(y4*dpsidfi+y2*dpsidr)/(sf*2.*pi)
#    y10=(y4*dpsidfi+y2*dpsidr)/sf
    y11=(y4*dpsidfi+y2*dpsidr)
    y12=-Etor*dRdtfi
    dydt=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12]
#    dydt=[y1,y2,y3,y4]
#    print(100000*R*R0*Etor/ccc,R,R0,Etor,ccc,(y4*dpsidfi+y2*dpsidr)/sf)
#    sys.exit()
#    if(r<0.):
#        print('r=',r,'t=',t,'E0tor=',E0tor)
#        print('thet=',thet,'thet/2pi=',thet/(2*pi))
#        print(y1,y2,y3,y4)
#        exit()
    return(dydt)


def eq_mot_1(eqq,ccc,m0,R0,pperp,ppar,r,thet,fi,R,Uloop,brtr,brtt,brtfi,gbr,gbt,gbfi, \
    bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi,brad,btor,bpol,muini,Btot,dBpoldr,dBpoldthet,dBpoldfi, \
    dBraddr,dBraddthet,dBraddfi,dBtordr,dBtordthet,dBtordfi,Bpol,Brad,Btor):
    E0tor=E0_field(r,thet,fi,R0,Uloop)
    Etot,Etor,etor,Erad,erad,Epol,epol=E_field(r,thet,fi,R0,E0tor)
    ptot2=pperp**2+ppar**2
    gam=sqrt(1.+ptot2)


#    dppardt=eqq*R0/(m0*ccc**2)*(Erad*brad+Epol*bpol+Etor*btor)-R0*pperp**2/(2*Btot*gam)* \
#    (gbr*brad+gbt*bpol+gbfi*btor)

    omce=eqq*Btot/(m0*ccc)
    M1=ppar*R0/gam
    M2=0.5*R0/(omce*gam)*pperp**2
#    M2=2*R0/(omce*gam)*pperp**2
    M3=R0/(omce*gam)*ppar**2
    M4=R0/(ccc*Btot)

#    dRdtr=M1*brad+M2*bgrr+M3*bbrtr+M4*(Epol*btor-Etor*bpol)
#    dRdtt=M1*bpol+M2*bgrt+M3*bbrtt+M4*(Etor*brad-Erad*btor)
    dRdtfi=M1*btor+M2*bgrfi+M3*bbrtfi+M4*(Erad*bpol-Epol*brad)
#    y1=dppardt
#    y2=dRdtr
#    y3=dRdtt/r
#    y4=dRdtfi/R
#    dpperp2dt=muini*(gbr*y2+gbt*y3*r+gbfi*y4*R)
#    y5=dpperp2dt
#    dBpoldt=dBpoldr*y2+dBpoldthet*y3+dBpoldfi*y4
#    y6=dBpoldt
#    dBtotdt=gbr*y2+gbt*y3*r+gbfi*y4*R
#    y7=dBtotdt
#    dBraddt=dBraddr*y2+dBraddthet*y3+dBraddfi*y4
#    y8=dBraddt
#    dBtordt=dBtordr*y2+dBtordthet*y3+dBtordfi*y4
#    y9=dBtordt
#    y10=R*R0*Etor/ccc+(y4*dpsidfi+y2*dpsidr)/(sf)
#    y11=R*R0*Etor/ccc+(y4*dpsidfi+y2*dpsidr)
#    dydt=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11]
    return(dRdtfi)

#import Splines 

def fin_fun(t,y,eqq,m0,ccc,a,R0,delr,delfi,nfi,n,pparini,pperpini,muini):
    ppar=y[0]
    r=y[1]
    thet=y[2]
    fi=y[3]
#    pperp2pr=y[4]
#    pperppr=sqrt(pperp2pr)
#    Bpol=y[5]
#    Btot=y[6]
#    Brad=y[7]
#    Btor=y[8]
    sf0=spl_q0(t)
    sfb=spl_qa(t)
#    sfb=Splines.spl_qa(t)
    Uloop=spl_U(t)
    B0=spl_B(t)
    sf=saf_fact(sf0,sfb,r,a,Uloop)
#    Bpol,Bpol1=Bpol_f(r,thet,sf,B0,R0)
#    E0tor=E0_field(r,thet,fi,R0,Uloop)
#    Etot,Etor,etor,Erad,erad,Epol,epol=E_field(r,thet,fi,R0,E0tor)
    R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1,psitor,dpsidr,dpsidfi,sf \
    =Mag_field(r,thet,fi,R0,a,B0,delfi,nfi,delr,n,sf0,sfb,Uloop)
    rtbr,rtbpol,rtbfi,brtr,brtt,brtfi,gbr,gbt,gbfi,bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi  \
    =rot_b(r,thet,fi,R,Btot,Btor,Bpol,Bpol1,Brad,brad,btor,bpol,bpol1,dBpoldr,   \
    dBtordfi,dBraddr,dBtordr,dBpoldfi,dBraddfi,  \
    dBpoldthet,dBtordthet,dBraddthet,dBpoldthet1,dBtordthet1,dBraddthet1)
    pperp2=muini*Btot
    pperp=sqrt(pperp2)
    dydt=eq_mot(t,eqq,ccc,m0,R0,pperp,ppar,r,thet,fi,R,Uloop,brtr,brtt,brtfi,gbr,gbt,gbfi, \
    bgrr,bgrt,bgrfi,bbrtr,bbrtt,bbrtfi,brad,btor,bpol,muini,Btot,dBpoldr,dBpoldthet,dBpoldfi,  \
    dBraddr,dBraddthet,dBraddfi,dBtordr,dBtordthet,dBtordfi,Bpol,Brad,Btor,psitor,dpsidr,dpsidfi,sf)
    return(dydt)




#B0=-2.2   # sign defines direction of toroidal drift, for B0<0 electron toroidal drift upword
#sf0=1.
#sfb=3.2





#import matplotlib.pyplot as plt


#plt.plot(tnp*tau_norm/ccc_R0, Uloopnp, 'g', label='Uloop')
#plt.plot(tnp1*tau_norm/ccc_R0, spl_U(tnp1), 'b', label='Uloop_spline')
#plt.legend(loc='best')
#plt.xlabel('t')
#plt.grid()
#plt.show()

#exit()

#exit()



#result_df = pd.DataFrame(columns=['pparnp', 'rnp', 'finp', 'thetnp', 'tnp1',])
#result_df = pd.DataFrame(columns=['pparini','rini','thetini','fiini','pperp2ini','Bpolini','Btotini','Bradini','Btorini','psipolini','psitorini',])
#rnp=np.linspace(0, a, num_it*nrange)
#Rmajnp=np.linspace(0, a, num_it*nrange)
#finp= np.linspace(0., 2*pi, num_it*nrange)
#thetnp= np.linspace(0., 2*pi, num_it*nrange)
#Btornp=np.linspace(0., 1., num_it*nrange)
#Btotnp=np.linspace(0., 1., num_it*nrange)
#Bradnp=np.linspace(0., 1., num_it*nrange)
#Bpolnp=np.linspace(0., 1., num_it*nrange)
#pparnp=np.linspace(0., 1., num_it*nrange)
#pperp2np=np.linspace(0., 1., num_it*nrange)
#tnp1=np.linspace(0., 1., num_it*nrange)
#psipolnp=np.linspace(0., 1., num_it*nrange)



#from parameters_FT2_r_3 import *
from parameters_EXL_50U_13976 import *
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
import os
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

num_it=2000
nrange=20000
delt=200000
#result_df = pd.DataFrame(columns=['pparnp', 'rnp', 'finp', 'thetnp', 'tnp1',])
result_df = pd.DataFrame(columns=['pparini','rini','thetini','fiini','pperp2ini','Bpolini','Btotini','Bradini','Btorini','psipolini','psitorini','energyini','t_ini',])

from scipy.integrate import odeint,solve_ivp    
for it in range(num_it):
    print(f"num it={it}")
    t0c=t_ini
    sf0=spl_q0(t0c)
    sfb=spl_qa(t0c)
    Uloop=spl_U(t0c)
    B0=spl_B(t0c)
    print('t_ini=',t0c,'sf0=',sf0,'sfb=',sfb,'B0=',B0,'Uloop=',Uloop)
    sf=saf_fact(sf0,sfb,rini,a,Uloop)
    print('rini=',rini,'thetini=',thetini,'fiini=',fiini,'pparini=',pparini)
    #sys.exit()
    #Bpol,Bpol1=Bpol_f(rini,thetini,sf,B0,R0) 
    #psipolini=pi*B0*a**2/(sfb-sf0)*log((sf0+(sfb-sf0)*(rini/a)**2)/sf0)
    



    y0=[pparini,rini,thetini,fiini,pperp2ini,Bpolini,Btotini,Bradini,Btorini,psipolini,psitorini,energyini]
    #y0=[pparini,rini,thetini,fiini]
    time=t_ini+delt  #t1UL
    print('rini=',rini,'thetini=',thetini,'fiini=',fiini,'pparini=',pparini,'energyini=',energyini)
    print('t_ini(s)=',t_ini*R0/ccc*tau_norm,'del_t_calculation(s)=',(time-t_ini)*R0/ccc*tau_norm,'time(s)=',time*R0/ccc*tau_norm)
   
    #from scipy.integrate.solve_ivp import odeint

    #sol= solve_ivp(fin_fun,  [0, time], y0, method='DOP853', args=(eqq,m0,ccc,a,R0,B0,sf0,sfb,delr,delfi,nfi,Uloop,n,pparini,pperpini,muini),  \
    #rtol = 1e-6, atol= 1e-9)
    #sol= solve_ivp(fin_fun,[0, time],y0,method='DOP853',t_eval= np.linspace(0., time, nrange), args=(eqq,m0,ccc,a,R0,B0,sf0,sfb,delr,delfi,nfi,Uloop,n,pparini,pperpini,muini)) 
    sol= solve_ivp(fin_fun,[t_ini, time],y0,method='DOP853',t_eval= np.linspace(t_ini, time, nrange),args=(eqq,m0,ccc,a,R0,delr,delfi,nfi,n,pparini,pperpini,muini),rtol = 1e-7, atol= 1e-10) 
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
#    print('thetini=',thetini,'fiini=',fiini)
    #    pperp2ini=sol.y[4,nrange-1]
    #    psipolini=sol.y[9,nrange-1]
    #    Bpolini=sol.y[5,nrange-1]
    #    Btorini=sol.y[8,nrange-1]
    #    Btotini=sol.y[6,nrange-1]
    #    Bradini=sol.y[7,nrange-1]
    #    psitorini=sol.y[10,nrange-1]
    #pparnp[it*nrange:nrange+it*nrange]=sol.y[0,0:nrange]
    #df = pd.DataFrame({
    #    'pparini': sol.y[0],
    #    'rnp'   : sol.y[1],
    #    'finp'  : sol.y[3], 
    #    'thetnp': sol.y[2],
    #    'tnp1'  : sol.t
    #    })
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
    print(result_df.head)
    result_df.to_pickle('result_11_equations_EXL_50U_13976_r_0.2_t_0.2_.pkl') 
#    df.to_pickle('final_data.pkl') 
#LSODA
#DOP853
# Сохраняем DataFrame в бинарный файл
#result_df.to_pickle('result.pkl') 