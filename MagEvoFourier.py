#
#
#                                        MagEvoFourier.py
#
#           Electron MHD at fixed temperature with Landau quantized fermions. In Cartesian coordinates with periodic domain.
#
#           Run in parallel using "mpirun -n X python3 MagEvoFourier.py" where X is the number of cores
#
#########################################################################################################

import re
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, special, optimize
from mpmath import mp

import h5py
import time
from IPython import display
from mpi4py import MPI

try:
    import cPickle as pickle
except ImportError:
    import pickle

import dedalus.public as d3
from dedalus.core.domain import Domain
import logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#########################Global variables##############################################################

c = 29979245800 #speed of light in cm/s
hbarc = 197.3269804 #hbar times c in MeV*fm
unit_e = 4.80320425e-10 # elementary charge in units of statcoulomb
yr = 3600*24*365 #1 year in seconds
G = 6.67430e-8 #Gravitational constant in dyn*cm^2/g^2
k_B = 8.61733034e-11 #Boltzmann constant in MeV/K
M_e = 0.51099895000 #electron mass in MeV
MeVtoErg = 1/6.2415e5 #conversion factor from MeV to erg
alpha_e = 0.0072973525693 #electromagnetic fine structure constant (dimensionless)
eB_crit = M_e**2 #critical magnetic field times elementary charge in MeV^2

GaussConverter = 6.241509074e-8*hbarc #Conversion factor between 1 statCoulomb*Gauss to __ MeV/fm*hbarc = ___ MeV^2
GaussConverter2 = 4.002719868e16/hbarc**1.5 #Conversion factor: 1 MeV^2 = 4.002719868e16/(197.3269804)^(3/2) G = 1.444027592e13 G (hbarc=c=1 all energy units)

B_0 = 1e15 #characteristic magnetic field (G)
n_e0 = 1e-4 #characteristic electron density (fm^{-3})
L_0 = 1e4 #characteristic length scale (cm)
tau = 4*mt.pi*unit_e*n_e0*1e39*L_0**2/(B_0*c) #characteristic timescale (s). Taken as the Hall time-scale with length scale 10 m and field 10^{15} G
T_0 = 1e8 #characteristic temperature (K)
E_0 = B_0*L_0/(c*tau) #characteristic electric field (statV/cm)

######################### Function definitions ##############################################################

def KroneckerDelta(n1,n2):
    #Only use n1, n2 being integers
    if abs(n1-n2)<0.5: KNret = 1
    else: KNret=0
    return KNret

def KroneckerDeltaArray(n1,n2):
    return np.where(np.abs(n1-n2)<0.5,1,0)

###################### Thermodynamic functions ##############################################################

#Computes thermodynamic derivatives Omega_xy but for simulations where T is held fixed
def Omega_xyFixedT(*args):
    Bfull = B_0*args[0].data #converts to G
    mufull = args[1].data
    M = args[2]
    Tfull = k_B*T_0*args[3].data #converts to MeV

    #Take 2D slice of B values for constant phi
    B = Bfull[0,:,:] 
    mu = mufull[0,:,:]
    T = Tfull[0,:,:]
    
    eB = B*unit_e*GaussConverter
    
    dPdB = np.zeros_like(eB)
    d2PdB2 = np.zeros_like(eB)
    d2PdBdmu = np.zeros_like(eB)

    n_max = np.floor( ( mu*mu-M*M )/(2*eB) ) #highest occupied Landau level at T=0

    mask = ( 2*np.pi**2*mu*T > eB ) 
    
    if(eB[mask].size != 0):
        dPdB[mask], d2PdB2[mask], d2PdBdmu[mask] = Omega_xyHighTFixedT(eB[mask],mu[mask],M,T[mask],n_max[mask])

    if(eB[~mask].size != 0):
        dPdB[~mask], d2PdB2[~mask], d2PdBdmu[~mask] = Omega_xyLowTFixedT(eB[~mask],mu[~mask],M,T[~mask],n_max[~mask])

    Mag = np.sqrt(alpha_e)*dPdB*GaussConverter2/B_0 #in MeV^2 -> G -> reduced units; equals -f_B
    d2PdB2 = alpha_e*d2PdB2
    d2PdBdmu = np.sqrt(alpha_e)*d2PdBdmu
    
    InstabilityMask = np.heaviside(1-4*np.pi*d2PdB2,0) #np.heaviside(-chi_n = f_BB > -1/(4.*np.pi),0)
    maskedd2PdB2 = ( -(1-4*np.pi*d2PdB2)*InstabilityMask + 1 )/(4*np.pi) #removes unstable regions (requires 1-4*np.pi*P_BB > 0)
    maskedOmegaBB = ( -maskedd2PdB2 ) #dimensionless; equals u_BB
    maskedOmegaBmu = ( -d2PdBdmu )*GaussConverter2/B_0 #in MeV -> G/MeV -> reduced units/MeV

    return np.expand_dims(Mag,axis=0), np.expand_dims(maskedOmegaBB,axis=0), np.expand_dims(maskedOmegaBmu,axis=0)#, np.expand_dims(f,axis=0)

#Computes second-order thermodynamic derivatives for low temperatures; used in Omega_xyFixedT
def Omega_xyLowTFixedT(eB,mu,M,T,n_max):

    T_B = 1*M*( np.sqrt(1+2*(n_max+1)*eB/eB_crit) - np.sqrt(1+2*n_max*eB/eB_crit) ) #critical temperature in MeV    

    npr = n_max - 1
    p_F = np.sqrt( mu*mu-M*M )
    E_Fnpr = np.sqrt( p_F*p_F-2*eB*npr )
    InvE_Fnpr = 1/E_Fnpr
    MagMnpr = np.sqrt( M*M+2*eB*npr )
    InvMagMnpr = 1/MagMnpr
    E_F1 = np.sqrt( p_F*p_F-2*eB )
    InvE_F1 = 1/E_F1
    MagM1 = np.sqrt( M*M+2*eB )    
    InvMagM1 = 1/MagM1
    B_2 = 1/6
    B_4 = -1/30
    
    dPdBTemp = ( 1/(4*np.pi**2)*( mu*p_F-M**2*np.log((mu+p_F)/M) ) 
                + 1/(2*np.pi**2)*( npr*(mu*E_Fnpr - MagMnpr**2*np.log((mu+E_Fnpr)*InvMagMnpr) )
                                      - (mu*E_F1 - MagM1**2*np.log((mu+E_F1)*InvMagM1)) )
                + 1/(4*np.pi**2)*( mu*E_Fnpr - (M**2+4*eB*npr)*np.log((mu+E_Fnpr)*InvMagMnpr) + mu*E_F1 - (M**2+4*eB)*np.log((mu+E_F1)*InvMagM1) )
                + B_2*eB/(2*np.pi**2)*( eB*npr*mu*InvMagMnpr**2*InvE_Fnpr - 2*np.log((mu+E_Fnpr)*InvMagMnpr)
                                              - eB*mu*InvMagM1**2*InvE_F1 + 2*np.log((mu+E_F1)*InvMagM1) )
                + B_4*eB**3*mu/(24*np.pi**2)*( ( 20*mu**2*(M**2+eB*npr)*MagMnpr**2 
                                                - 3*(4*M**2+3*eB*npr)*MagMnpr**4 
                                                - 8*mu**4*(M**2+eB*npr) )*InvMagMnpr**6*InvE_Fnpr**5
                                              - ( 20*mu**2*(M**2+eB)*MagM1**2 
                                                - 3*(4*M**2+3*eB)*MagM1**4 
                                                - 8*mu**4*(M**2+eB) )*InvMagM1**6*InvE_F1**5 )
                + mu*T**2/6*( 0.5*(mu**2-M**2)/p_F**3
                                             + ( npr*InvE_Fnpr - InvE_F1 )
                                             + 0.5*( (mu**2-M**2-eB*npr)*InvE_Fnpr**3 + (mu**2-M**2-eB)*InvE_F1**3)
                                             + B_2*eB/2*( (2*(mu**2-M**2)-eB*npr)*InvE_Fnpr**5 - (2*(mu**2-M**2)-eB)*InvE_F1**5 ) ) )
    d2PdB2Temp = ( 1/(np.pi**2)*( -npr**2*np.log((mu+E_Fnpr)*InvMagMnpr) + np.log((mu+E_F1)*InvMagM1) )
                         + 1/(2*np.pi**2)*( npr*( eB*npr*mu*InvMagMnpr**2*InvE_Fnpr - 2*np.log((mu+E_Fnpr)*InvMagMnpr) )
                                              + ( eB*mu*InvMagM1**2*InvMagM1 - 2*np.log((mu+E_F1)*InvMagM1) ) )
                         + B_2/(2*np.pi**2)*( -eB*npr*mu*( 4*M**4 + M**2*(13*eB*npr-4*mu**2)+2*eB*npr*(5*eB*npr-3*mu**2) )*InvMagMnpr**4*InvE_Fnpr**3
                                                  - 2*np.log((mu+E_Fnpr)*InvMagMnpr)
                                            + eB*mu*( 4*M**4 + M**2*(13*eB-4*mu**2)+2*eB*(5*eB-3*mu**2) )*InvMagM1**4*InvE_F1**3
                                                  + 2*np.log((mu+E_F1)*InvMagM1) )
                         + B_4*eB**2*mu/(24*np.pi**2)*( -( 2*mu**2*MagMnpr**4*(48*M**4+52*M**2*eB*npr+17*eB**2*npr**2)
                                                                         - 3*MagMnpr**6*(12*M**4+8*M**2*eB*npr+3*eB**2*npr**2)
                                                                         + 8*mu**6*(3*M**4+4*M**2*eB*npr+2*eB**2*npr**2)
                                                                         - 28*mu**4*MagMnpr**2*(3*M**4+4*M**2*eB*npr+2*eB**2*npr**2) )*InvMagMnpr**8*InvE_Fnpr**7
                                                                  + ( 2*mu**2*MagM1**4*(48*M**4+52*M**2*eB+17*eB**2)
                                                                         - 3*MagM1**6*(12*M**4+8*M**2*eB+3*eB**2)
                                                                         + 8*mu**6*(3*M**4+4*M**2*eB+2*eB**2)
                                                                         - 28*mu**4*MagM1**2*(3*M**4+4*M**2*eB+2*eB**2) )*InvMagM1**8*InvE_F1**7 )
                         + T**2*mu/6*( (npr*2*InvE_Fnpr**3 - InvE_F1**3) + 0.5*( (2*(mu**2-M**2)-eB*npr)*InvE_Fnpr**5*npr + (2*(mu**2-M**2)-eB)*InvE_F1**5 )
                                                      + B_2/2*( ( 2*M**4-(eB*npr)**2+4*eB*npr*mu**2+2*mu**4-4*M**2*(eB*npr+mu**2) )*InvE_Fnpr**7 
                                                               - ( 2*M**4-(eB)**2+4*eB*mu**2+2*mu**4-4*M**2*(eB+mu**2) )*InvE_F1**7) ) )
    d2PdBdmuTemp = ( p_F/(2*np.pi**2) + 1/(np.pi*np.pi)*( npr*E_Fnpr - E_F1 )
                    + 1/(2*np.pi**2)*( ( p_F*p_F-3*eB*npr )*InvE_Fnpr + ( p_F*p_F-3*eB )*InvE_F1 )
                    + B_2*eB/(2*np.pi**2)*( -( 2*p_F*p_F-3*eB*npr )*InvE_Fnpr**3 + ( 2*p_F*p_F-3*eB )*InvE_F1**3 )
                    + B_4*eB**3/(8*np.pi**2)*( -( 4*p_F*p_F-3*eB*npr )*InvE_Fnpr**7 + ( 4*p_F*p_F-3*eB )*InvE_F1**7 ) 
                    - T**2/6*( 0.5*M**2*(mu**2-M**2)/(p_F**2-M**2)**2.5 + npr*MagMnpr*InvE_Fnpr**3 - MagM1*InvE_F1**3
                              + 0.5*( (mu**2*(M**2+4*eB*npr) - MagMnpr**2*(M**2+eB*npr))*InvE_Fnpr**5 + (mu**2*(M**2+4*eB) - MagM1**2*(M**2+eB))*InvE_F1**5 ) 
                              + B_2*eB/2*( ( 4*mu**4 - 2*mu**2*(M**2-4*eB*npr) - 2*(eB*npr)**2 -5*eB*npr*M**2 - 2*M**4 )*InvE_Fnpr**7 
                                          - ( 4*mu**4 - 2*mu**2*(M**2-4*eB) - 2*eB**2 -5*eB*M**2 - 2*M**4 )*InvE_F1**7) ) )
    
    dPdB0 = 1/(2*np.pi**2)*( 0.5*mu*p_F - 0.5*M**2*np.log( (mu+p_F)/M ) + np.pi**2/6*mu/p_F*T**2 + 7*np.pi**4/120*mu*M**2/p_F**5*T**4 )
    d2PdB20 = 0
    d2PdBdmu0 = 1/(2*np.pi**2)*( p_F - np.pi**2/6*mu*M**2/p_F**3*T**2 - 7*np.pi**4/120*M**2*(4*mu**2+M**2)/p_F**7*T**4 )
    
    conditionsdPdB = [ n_max == 0, n_max == 1 ]
    choicesdPdB = [ 0, dPdB0 ]
    dPdB = np.select(conditionsdPdB,choicesdPdB,dPdBTemp)
    conditionsd2PdB2 = [ n_max == 0, n_max == 1 ]
    choicesd2PdB2 = [ 0, d2PdB20 ]
    d2PdB2 = np.select(conditionsd2PdB2,choicesd2PdB2,d2PdB2Temp)
    conditionsd2PdBdmu = [ n_max == 0, n_max == 1 ]
    choicesd2PdBdmu = [ 0, d2PdBdmu0 ]
    d2PdBdmu = np.select(conditionsd2PdBdmu,choicesd2PdBdmu,d2PdBdmuTemp)
    
    m_n = np.sqrt(M**2+2*eB*n_max) #m_{n_max}
    m_np1 = np.sqrt(M**2+2*eB*(n_max+1)) #m_{n_max+1}
    
    a = m_n/T
    b = mu/T
    
    mask = ( np.logical_and( np.abs(b - a) < 50, np.sqrt(a**2+b**2) > 50) )
    G1_ab = np.empty_like(eB)
    G2_ab = np.empty_like(eB)
    H2_ab = np.empty_like(eB)
    I1_ab = np.empty_like(eB)
    I2_ab = np.empty_like(eB)
    conditions_a = [ b - a > 50 ]
    choices_G1_a = [ b*np.sqrt( np.abs((b/a)**2-1) )/(2*a) + 0.5*np.log( np.sqrt( np.abs((b/a)**2-1) ) + b/a ) + np.pi**2*b*(b**2-2*a**2)/(6*a**2*(np.abs(b**2-a**2))**1.5) - 7*np.pi**4/120*b*(4*a**2+b**2)/(np.abs(b**2-a**2))**3.5 ]
    choices_G2_a = [ np.log( np.sqrt( np.abs((b/a)**2-1) ) + b/a ) - np.pi**2*b/(6*(np.abs(b**2-a**2))**1.5) - 7*np.pi**4/120*b*(2*b**2+3*a**2)/(np.abs(b**2-a**2))**3.5 ]
    choices_H2_a = [ -b/np.sqrt( np.abs(b**2-a**2) ) - np.pi**2*b*a**2/(2*(np.abs(b**2-a**2))**2.5) - 7*np.pi**4/24*b*a**2*(4*b**2+3*a**2)/(np.abs(b**2-a**2))**4.5 ]
    choices_I1_a = [ np.sqrt( np.abs(b**2/a**2-1) ) - np.pi**2*a/(6*(np.abs(b**2-a**2))**1.5) - 7*np.pi**4/120*a*(4*b**2+3*a**2)/(np.abs(b**2-a**2))**3.5 ]
    choices_I2_a = [ -a/np.sqrt( np.abs(b**2-a**2) ) - np.pi**2*a*(2*b**2+a**2)/(6*(np.abs(b**2-a**2))**2.5) - 7*np.pi**4/120*a*(8*b**4+24*a**2*b**2+3*a**4)/(np.abs(b**2-a**2))**4.5 ]
    G1_ab = np.select(conditions_a,choices_G1_a, np.sqrt(np.pi/2)*np.exp(b-a)*( 1/a**0.5 + 7/(8*a**1.5) + 57/(128*a**2.5) - 195/(1024*a**3.5) ) )
    G2_ab = np.select(conditions_a,choices_G2_a, np.sqrt(np.pi/2)*np.exp(b-a)*( 1/a**0.5 - 1/(8*a**1.5) + 9/(128*a**2.5) - 75/(1024*a**3.5) ) )
    H2_ab = np.select(conditions_a,choices_H2_a, -np.sqrt(np.pi/2)*np.exp(b-a)*( a**0.5 + 3/(8*a**0.5) - 15/(128*a**1.5) + 105/(1024*a**2.5) ) )
    I1_ab = np.select(conditions_a,choices_I1_a, np.sqrt(np.pi/2)*np.exp(b-a)*( 1/a**0.5 + 3/(8*a**1.5) - 15/(128*a**2.5) + 105/(1024*a**3.5) ) )
    I2_ab = np.select(conditions_a,choices_I2_a, -np.sqrt(np.pi/2)*np.exp(b-a)*( a**0.5 - 1/(8*a**0.5) + 9/(128*a**1.5) - 75/(1024*a**2.5) ) )
    G1_ab[mask] = -np.sqrt(np.pi/2)*( 1/a[mask]**0.5*PolyLog1_2Interp( b[mask] - a[mask] ) + 7/(8*a[mask]**1.5)*PolyLog3_2Interp( b[mask] - a[mask] ) + 57/(128*a[mask]**2.5)*PolyLog5_2Interp( b[mask] - a[mask] ) - 195/(1024*a[mask]**3.5)*PolyLog7_2Interp( b[mask] - a[mask] ) )
    G2_ab[mask] = -np.sqrt(np.pi/2)*( 1/a[mask]**0.5*PolyLog1_2Interp( b[mask] - a[mask] ) - 1/(8*a[mask]**1.5)*PolyLog3_2Interp( b[mask] - a[mask] ) + 9/(128*a[mask]**2.5)*PolyLog5_2Interp( b[mask] - a[mask] ) - 75/(1024*a[mask]**3.5)*PolyLog7_2Interp( b[mask] - a[mask] ) )
    H2_ab[mask] = np.sqrt(np.pi/2)*( a[mask]**0.5*PolyLogN1_2Interp( b[mask] - a[mask] ) + 3/(8*a[mask]**0.5)*PolyLog1_2Interp( b[mask] - a[mask] ) - 15/(128*a[mask]**1.5)*PolyLog3_2Interp( b[mask] - a[mask] ) + 105/(1024*a[mask]**2.5)*PolyLog5_2Interp( b[mask] - a[mask] ) )
    I1_ab[mask] = -np.sqrt(np.pi/2)*( 1/a[mask]**0.5*PolyLog1_2Interp( b[mask] - a[mask] ) + 3/(8*a[mask]**1.5)*PolyLog3_2Interp( b[mask] - a[mask] ) - 15/(128*a[mask]**2.5)*PolyLog5_2Interp( b[mask] - a[mask] ) + 105/(1024*a[mask]**3.5)*PolyLog7_2Interp( b[mask] - a[mask] ) )
    I2_ab[mask] = np.sqrt(np.pi/2)*( a[mask]**0.5*PolyLogN1_2Interp( b[mask] - a[mask] ) - 1/(8*a[mask]**0.5)*PolyLog1_2Interp( b[mask] - a[mask] ) + 9/(128*a[mask]**1.5)*PolyLog3_2Interp( b[mask] - a[mask] ) - 75/(1024*a[mask]**2.5)*PolyLog5_2Interp( b[mask] - a[mask] ) )
    conditions_b = [ np.sqrt(a**2+b**2) < 50  ]
    choices_G1_b = [ np.exp(interpolate._dfitpack.bispeu(LogG1Interp.tck[0],LogG1Interp.tck[1],LogG1Interp.tck[2],LogG1Interp.degrees[0],LogG1Interp.degrees[1],a,b)[0]) ]
    choices_G2_b = [ np.exp(interpolate._dfitpack.bispeu(LogG2Interp.tck[0],LogG2Interp.tck[1],LogG2Interp.tck[2],LogG2Interp.degrees[0],LogG2Interp.degrees[1],a,b)[0]) ]
    choices_H2_b = [ -np.exp(interpolate._dfitpack.bispeu(LogNH2Interp.tck[0],LogNH2Interp.tck[1],LogNH2Interp.tck[2],LogNH2Interp.degrees[0],LogNH2Interp.degrees[1],a,b)[0]) ]
    choices_I1_b = [ np.exp(interpolate._dfitpack.bispeu(LogI1Interp.tck[0],LogI1Interp.tck[1],LogI1Interp.tck[2],LogI1Interp.degrees[0],LogI1Interp.degrees[1],a,b)[0]) ]
    choices_I2_b = [ -np.exp(interpolate._dfitpack.bispeu(LogNI2Interp.tck[0],LogNI2Interp.tck[1],LogNI2Interp.tck[2],LogNI2Interp.degrees[0],LogNI2Interp.degrees[1],a,b)[0]) ]
    G1_ab = np.select(conditions_b, choices_G1_b, G1_ab )
    G2_ab = np.select(conditions_b, choices_G2_b, G2_ab )
    H2_ab = np.select(conditions_b, choices_H2_b, H2_ab )
    I1_ab = np.select(conditions_b, choices_I1_b, I1_ab )
    I2_ab = np.select(conditions_b, choices_I2_b, I2_ab )
    
    a1 = m_np1/T
    mask1 = ( np.logical_and(np.abs(b - a1) < 50, np.sqrt(a1**2+b**2) > 50) )
    G1_ab1 = np.empty_like(eB)
    G2_ab1 = np.empty_like(eB)
    H2_ab1 = np.empty_like(eB)
    I1_ab1 = np.empty_like(eB)
    I2_ab1 = np.empty_like(eB)
    conditions1_a = [ b - a1 > 50 ]
    choices1_G1_a = [ b*np.sqrt( np.abs((b/a1)**2-1) )/(2*a1) + 0.5*np.log( np.sqrt( np.abs((b/a1)**2-1) ) + b/a1 ) + np.pi**2*b*(2*b**2-a1**2)/(6*a1**2*(np.abs(b**2-a1**2))**1.5) - 7*np.pi**4/120*b*(4*a1**2+b**2)/(np.abs(b**2-a1**2))**3.5 ]
    choices1_G2_a = [ np.log( np.sqrt( np.abs((b/a1)**2-1) ) + b/a1 ) - np.pi**2*b/(6*(np.abs(b**2-a1**2))**1.5) - 7*np.pi**4/120*b*(2*b**2+3*a1**2)/(np.abs(b**2-a1**2))**3.5 ]
    choices1_H2_a = [ -b/np.sqrt( np.abs(b**2-a1**2) ) - np.pi**2*b*a1**2/(2*(np.abs(b**2-a1**2))**2.5) - 7*np.pi**4/24*b*a1**2*(4*b**2+3*a1**2)/(np.abs(b**2-a1**2))**4.5 ]
    choices1_I1_a = [ np.sqrt( np.abs(b**2/a1**2-1) ) - np.pi**2*a1/(6*(np.abs(b**2-a1**2))**1.5) - 7*np.pi**4/120*a1*(4*b**2+3*a1**2)/(np.abs(b**2-a1**2))**3.5 ]
    choices1_I2_a = [ -a1/np.sqrt( np.abs(b**2-a1**2) ) - np.pi**2*a1*(2*b**2+a1**2)/(6*(np.abs(b**2-a1**2))**2.5) - 7*np.pi**4/120*a1*(8*b**4+24*a1**2*b**2+3*a1**4)/(np.abs(b**2-a1**2))**4.5 ]
    G1_ab1 = np.select(conditions1_a,choices1_G1_a, np.sqrt(np.pi/2)*np.exp(b-a1)*( 1/a1**0.5 + 7/(8*a1**1.5) + 57/(128*a1**2.5) - 195/(1024*a1**3.5) ) )
    G2_ab1 = np.select(conditions1_a,choices1_G2_a, np.sqrt(np.pi/2)*np.exp(b-a1)*( 1/a1**0.5 - 1/(8*a1**1.5) + 9/(128*a1**2.5) - 75/(1024*a1**3.5) ) )
    H2_ab1 = np.select(conditions1_a,choices1_H2_a, -np.sqrt(np.pi/2)*np.exp(b-a1)*( a1**0.5 + 3/(8*a1**0.5) - 15/(128*a1**1.5) + 105/(1024*a1**2.5) ) )
    I1_ab1 = np.select(conditions1_a,choices1_I1_a, np.sqrt(np.pi/2)*np.exp(b-a1)*( 1/a1**0.5 + 3/(8*a1**1.5) - 15/(128*a1**2.5) + 105/(1024*a1**3.5) ) )
    I2_ab1 = np.select(conditions1_a,choices1_I2_a, -np.sqrt(np.pi/2)*np.exp(b-a1)*( a1**0.5 - 1/(8*a1**0.5) + 9/(128*a1**1.5) - 75/(1024*a1**2.5) ) )
    G1_ab1[mask1] = -np.sqrt(np.pi/2)*( 1/a1[mask1]**0.5*PolyLog1_2Interp( b[mask1] - a1[mask1] ) + 7/(8*a1[mask1]**1.5)*PolyLog3_2Interp( b[mask1] - a1[mask1] ) + 57/(128*a1[mask1]**2.5)*PolyLog5_2Interp( b[mask1] - a1[mask1] ) - 195/(1024*a1[mask1]**3.5)*PolyLog7_2Interp( b[mask1] - a1[mask1] ) )
    G2_ab1[mask1] = -np.sqrt(np.pi/2)*( 1/a1[mask1]**0.5*PolyLog1_2Interp( b[mask1] - a1[mask1] ) - 1/(8*a1[mask1]**1.5)*PolyLog3_2Interp( b[mask1] - a1[mask1] ) + 9/(128*a1[mask1]**2.5)*PolyLog5_2Interp( b[mask1] - a1[mask1] ) - 75/(1024*a1[mask1]**3.5)*PolyLog7_2Interp( b[mask1] - a1[mask1] ) )
    H2_ab1[mask1] = np.sqrt(np.pi/2)*( a1[mask1]**0.5*PolyLogN1_2Interp( b[mask1] - a1[mask1] ) + 3/(8*a1[mask1]**0.5)*PolyLog1_2Interp( b[mask1] - a1[mask1] ) - 15/(128*a1[mask1]**1.5)*PolyLog3_2Interp( b[mask1] - a1[mask1] ) + 105/(1024*a1[mask1]**2.5)*PolyLog5_2Interp( b[mask1] - a1[mask1] ) )
    I1_ab1[mask1] = -np.sqrt(np.pi/2)*( 1/a1[mask1]**0.5*PolyLog1_2Interp( b[mask1] - a1[mask1] ) + 3/(8*a1[mask1]**1.5)*PolyLog3_2Interp( b[mask1] - a1[mask1] ) - 15/(128*a1[mask1]**2.5)*PolyLog5_2Interp( b[mask1] - a1[mask1] ) + 105/(1024*a1[mask1]**3.5)*PolyLog7_2Interp( b[mask1] - a1[mask1] ) )
    I2_ab1[mask1] = np.sqrt(np.pi/2)*( a1[mask1]**0.5*PolyLogN1_2Interp( b[mask1] - a1[mask1] ) - 1/(8*a1[mask1]**0.5)*PolyLog1_2Interp( b[mask1] - a1[mask1] ) + 9/(128*a1[mask1]**1.5)*PolyLog3_2Interp( b[mask1] - a1[mask1] ) - 75/(1024*a1[mask1]**2.5)*PolyLog5_2Interp( b[mask1] - a1[mask1] ) )
    conditions1_b = [ np.sqrt(a1**2+b**2) < 50  ]
    choices1_G1_b = [ np.exp(interpolate._dfitpack.bispeu(LogG1Interp.tck[0],LogG1Interp.tck[1],LogG1Interp.tck[2],LogG1Interp.degrees[0],LogG1Interp.degrees[1],a1,b)[0]) ]
    choices1_G2_b = [ np.exp(interpolate._dfitpack.bispeu(LogG2Interp.tck[0],LogG2Interp.tck[1],LogG2Interp.tck[2],LogG2Interp.degrees[0],LogG2Interp.degrees[1],a1,b)[0]) ]
    choices1_H2_b = [ -np.exp(interpolate._dfitpack.bispeu(LogNH2Interp.tck[0],LogNH2Interp.tck[1],LogNH2Interp.tck[2],LogNH2Interp.degrees[0],LogNH2Interp.degrees[1],a1,b)[0]) ]
    choices1_I1_b = [ np.exp(interpolate._dfitpack.bispeu(LogI1Interp.tck[0],LogI1Interp.tck[1],LogI1Interp.tck[2],LogI1Interp.degrees[0],LogI1Interp.degrees[1],a1,b)[0]) ]
    choices1_I2_b = [ -np.exp(interpolate._dfitpack.bispeu(LogNI2Interp.tck[0],LogNI2Interp.tck[1],LogNI2Interp.tck[2],LogNI2Interp.degrees[0],LogNI2Interp.degrees[1],a1,b)[0]) ]
    G1_ab1 = np.select(conditions1_b, choices1_G1_b, G1_ab1 )
    G2_ab1 = np.select(conditions1_b, choices1_G2_b, G2_ab1 )
    H2_ab1 = np.select(conditions1_b, choices1_H2_b, H2_ab1 )
    I1_ab1 = np.select(conditions1_b, choices1_I1_b, I1_ab1 )
    I2_ab1 = np.select(conditions1_b, choices1_I2_b, I2_ab1 )
    
    dPdB = dPdB + (2-KroneckerDeltaArray(n_max,0))/(2*np.pi**2)*( m_n**2*G1_ab - (m_n**2+eB*n_max)*G2_ab )
    dPdB = dPdB + 1/(np.pi**2)*( m_np1**2*G1_ab1 - (m_np1**2+eB*(n_max+1))*G2_ab1 )
    d2PdB2 = d2PdB2 - n_max/(2*np.pi**2)*(2-KroneckerDeltaArray(n_max,0))*( 2*G2_ab + eB*n_max/m_n**2*H2_ab )
    d2PdB2 = d2PdB2 - (n_max+1)/(np.pi**2)*( 2*G2_ab1 + eB*(n_max+1)/m_np1**2*H2_ab1 )
    d2PdBdmu = d2PdBdmu + (2-KroneckerDeltaArray(n_max,0))/(2*np.pi**2)*m_n*( I1_ab + eB*n_max/m_n**2*I2_ab )
    d2PdBdmu = d2PdBdmu + 1/(np.pi**2)*m_np1*( I1_ab1 + eB*(n_max+1)/m_np1**2*I2_ab1 )

    return dPdB*np.heaviside(T_B-T,0), d2PdB2*np.heaviside(T_B-T,0), d2PdBdmu*np.heaviside(T_B-T,0)#, ( d2Pdmu2*np.heaviside(T_B-T,0) + mu*np.sqrt(mu*mu-M*M)/(np.pi**2)*np.heaviside(T-T_B,1) )#, f

#Computes second-order thermodynamic derivatives for high temperatures; used in Omega_xyFixedT
def Omega_xyHighTFixedT(eB,mu,M,T,n_max):

    T_B = 100*M*( np.sqrt(1+2*(n_max+1)*eB/eB_crit) - np.sqrt(1+2*n_max*eB/eB_crit) ) #critical temperature in MeV  
          
    dPdB = np.zeros_like(eB)
    d2PdB2 = np.zeros_like(eB)
    d2PdBdmu = np.zeros_like(eB)

    i1val = i1Interp((mu**2-M**2)/eB)
    i2val = i2Interp((mu**2-M**2)/eB)
    
    h1 = interpolate._dfitpack.bispeu(h1Interp.tck[0],h1Interp.tck[1],h1Interp.tck[2],h1Interp.degrees[0],h1Interp.degrees[1],mu/M,eB/M**2)[0]
    h2 = interpolate._dfitpack.bispeu(h2Interp.tck[0],h2Interp.tck[1],h2Interp.tck[2],h2Interp.degrees[1],h2Interp.degrees[1],mu/M,eB/M**2)[0]
    
    PBReg = ( eB**0.5/(2*np.pi**2.5)*h1 - M**2/(4*np.pi**2.5*eB**0.5)*h2
             - 1/(8*np.pi**2.5)*eB**0.5*( mu*i1val - np.sqrt(2/np.pi)*M*special.zeta(1.5) ) )
    PBBReg = ( 1/(2*np.pi**2.5*eB**0.5)*h1 - M**2/(2*np.pi**2.5*eB**1.5)*h2
             - 5/(16*np.pi**2.5*np.sqrt(eB))*( mu*i1val - np.sqrt(2/np.pi)*M*special.zeta(1.5) ) )
    PBmuReg = 3*eB**0.5/(8*np.pi**2.5)*i1val + (mu**2-M**2)/(4*np.pi**2.5*eB**0.5)*i2val

    PBOsc = np.zeros_like(eB)
    PBBOsc = np.zeros_like(eB)
    PBmuOsc = np.zeros_like(eB)
    R_T = np.zeros_like(eB)
    
    for n in range(1,6):
        lamb = 2*np.pi**2*T*mu*n/eB
             
        mask = ( lamb < 10. )
        R_T[mask] = lamb[mask]/np.sinh(lamb[mask])
        R_T[~mask] = 2*lamb[~mask]*np.exp(-lamb[~mask])

        Sin = np.sin( np.pi/4 - np.pi*n/eB*(mu**2-M**2) )
        Cos = np.cos( np.pi/4 - np.pi*n/eB*(mu**2-M**2) )
        
        PBOsc = PBOsc + R_T*( eB**0.5*(mu**2-M**2)/(4*np.pi**3*mu*n**1.5)*Sin - 5*eB**1.5/(8*np.pi**4*mu*n**2.5)*Cos ) 
        PBBOsc = PBBOsc + R_T*( (mu**2-M**2)**2/(4*np.pi**2*mu*eB**1.5*np.sqrt(n))*Cos + 3*(mu**2-M**2)/(4*np.pi**3*mu*np.sqrt(eB)*n**1.5)*Sin )
        PBmuOsc = PBmuOsc - R_T*( 3*eB**0.5/(4*np.pi**3*n**1.5)*Sin + (mu**2-M**2)/(2*np.pi**2*np.sqrt(eB*n))*Cos )
           
    dPdB = ( PBOsc + PBReg )*np.heaviside(T_B-T,0)
    d2PdB2 = ( PBBOsc + PBBReg )*np.heaviside(T_B-T,0)
    d2PdBdmu = ( PBmuOsc + PBmuReg )*np.heaviside(T_B-T,0)
   
    return dPdB, d2PdB2, d2PdBdmu


#Computes electron number density in units of fm^{-3} with finite T. Arguments: B in G, mu in MeV, M in MeV, T in MeV, additional Landau levels to include above n_max
def n_eCalc(*args):
    
    Bfull = B_0*args[0].data #converts to G
    mufull = args[1].data
    M = args[2]
    LandauQuantThermo = args[4]
    
    B = Bfull[0,:,:]
    mu = mufull[0,:,:]
    
    eB = B*unit_e*GaussConverter

    n_max = np.floor((mu**2-M**2)/(2*eB))
    npr = n_max - 1
    E_Fnpr = np.sqrt( np.abs(mu**2-M**2-2*eB*npr) )
    InvE_Fnpr = 1/E_Fnpr
    E_F1 = np.sqrt( np.abs(mu**2-M**2-2*eB) )
    InvE_F1 = 1/E_F1
    B_2 = 1/6
    B_4 = -1/30

    if LandauQuantThermo == 1:
        n_e0 = (mu**2 - M**2)**1.5/(3*np.pi**2) #+ (2*mu**2-M**2)/(6*np.sqrt(mu**2-M**2))*T**2 #n_e at B=0 in MeV^3
    
        mask = ( n_max < 200 ) #returns n_e0 if n_max < 200 i.e., uses the B=0 result for n_e in this case
    
        n_e = np.zeros_like(eB)
        n_e[mask] = ( eB[mask]/(2*np.pi**2)*( np.sqrt(np.abs(mu[mask]**2-M**2)) + 2*np.sqrt(np.abs(mu[mask]**2-M**2-2*eB[mask]*n_max[mask]))*np.heaviside(n_max[mask]-0.5,0)
                               + 2*( -1/(3*eB[mask])*( E_Fnpr[mask]**3 - E_F1[mask]**3 ) + 0.5*( E_Fnpr[mask] + E_F1[mask] ) - 0.5*B_2*eB[mask]*( InvE_Fnpr[mask] - InvE_F1[mask] ) - 0.125*B_4*eB[mask]**3*( InvE_Fnpr[mask]**5 - InvE_F1[mask]**5 ) )*np.heaviside( n_max[mask]-1.5,0 ) ) )
        n_e[~mask] = n_e0[~mask]
    else:
        n_e = (mu**2 - M**2)**1.5/(3*np.pi**2) #n_e at B=0 in MeV^3

    return np.expand_dims(n_e,axis=0)/hbarc**3


######################### Interpolating functions used in thermodynamic functions #####################################################

#Create interpolating functions for J_0--J_2, dJ_0/dY--dJ_5/dY, I_0--I_1 and H_0 used in partial derivatives of grand potential density

YinterpMaxN = -45
YinterpMaxP = 45
Y_array = np.linspace(YinterpMaxN-1,YinterpMaxP+1,num=300,endpoint=True)

J_0Integral = []
dJ_0dYIntegral = []
J_1Integral = []
dJ_1dYIntegral = []
J_2Integral = []
dJ_2dYIntegral = []
dJ_3dYIntegral = []
dJ_4dYIntegral = []
I_0Integral = []
I_1Integral = []
H_0Integral = []

def J_0Integrand(y,Y):
    if y > 2e2:
        return np.sqrt(y+Y)*np.exp(-y)
    else:
        return np.sqrt(y+Y)/(np.exp(y)+2+np.exp(-y))
def dJ_0dYIntegrand(y,Y):
    if y > 2e2:
        return 1/(2*np.sqrt(y+Y))*np.exp(-y)
    else:
        return 1/(2*np.sqrt(y+Y))/(np.exp(y)+2+np.exp(-y))
def J_1Integrand(y,Y):
    if y > 2e2:
        return y*np.sqrt(y+Y)*np.exp(-y)
    else:    
        return y*np.sqrt(y+Y)/(np.exp(y)+2+np.exp(-y))
def dJ_1dYIntegrand(y,Y):
    if y > 2e2:
        return y/(2*np.sqrt(y+Y))*np.exp(-y)
    else:        
        return y/(2*np.sqrt(y+Y))/(np.exp(y)+2+np.exp(-y))
def J_2Integrand(y,Y):
    if y > 2e2:
        return y**2*np.sqrt(y+Y)*np.exp(-y)
    else:       
        return y**2*np.sqrt(y+Y)/(np.exp(y)+2+np.exp(-y))
def dJ_2dYIntegrand(y,Y):
    if y > 2e2:
        return y**2/(2*np.sqrt(y+Y))*np.exp(-y)
    else:       
        return y**2/(2*np.sqrt(y+Y))/(np.exp(y)+2+np.exp(-y))
def dJ_3dYIntegrand(y,Y):
    if y > 2e2:
        return y**3/(2*np.sqrt(y+Y))*np.exp(-y)
    else:         
        return y**3/(2*np.sqrt(y+Y))/(np.exp(y)+2+np.exp(-y))
def dJ_4dYIntegrand(y,Y):
    if y > 2e2:
        return y**4/(2*np.sqrt(y+Y))*np.exp(-y)
    else:         
        return y**4/(2*np.sqrt(y+Y))/(np.exp(y)+2+np.exp(-y))
def I_0Integrand(y,Y):
    if y > 2e2:
        return (y+Y)**1.5*np.exp(-y)
    else:     
        return (y+Y)**1.5/(np.exp(y)+2+np.exp(-y))
def I_1Integrand(y,Y):
    if y > 2e2:
        return y*(y+Y)**1.5*np.exp(-y)
    else:      
        return y*(y+Y)**1.5/(np.exp(y)+2+np.exp(-y))
def H_0Integrand(y,Y):
    if y > 2e2:
        return (y+Y)**2.5*np.exp(-y)
    else:      
        return (y+Y)**2.5/(np.exp(y)+2+np.exp(-y))  
     
for Y in Y_array:
    J_0Integral.append(integrate.quad(J_0Integrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    dJ_0dYIntegral.append(integrate.quad(dJ_0dYIntegrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    J_1Integral.append(integrate.quad(J_1Integrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    dJ_1dYIntegral.append(integrate.quad(dJ_1dYIntegrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    J_2Integral.append(integrate.quad(J_2Integrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    dJ_2dYIntegral.append(integrate.quad(dJ_2dYIntegrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    dJ_3dYIntegral.append(integrate.quad(dJ_3dYIntegrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    dJ_4dYIntegral.append(integrate.quad(dJ_4dYIntegrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    I_0Integral.append(integrate.quad(I_0Integrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])
    I_1Integral.append(integrate.quad(I_1Integrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])    
    H_0Integral.append(integrate.quad(H_0Integrand, -Y, np.inf, args=(Y),epsrel=1e-10)[0])    

J_0Interp = interpolate.interp1d(Y_array,J_0Integral,kind='cubic',fill_value='extrapolate')
dJ_0dYInterp = interpolate.interp1d(Y_array,dJ_0dYIntegral,kind='cubic',fill_value='extrapolate')
J_1Interp = interpolate.interp1d(Y_array,J_1Integral,kind='cubic',fill_value='extrapolate')
dJ_1dYInterp = interpolate.interp1d(Y_array,dJ_1dYIntegral,kind='cubic',fill_value='extrapolate')
J_2Interp = interpolate.interp1d(Y_array,J_2Integral,kind='cubic',fill_value='extrapolate')
dJ_2dYInterp = interpolate.interp1d(Y_array,dJ_2dYIntegral,kind='cubic',fill_value='extrapolate')
dJ_3dYInterp = interpolate.interp1d(Y_array,dJ_3dYIntegral,kind='cubic',fill_value='extrapolate')
dJ_4dYInterp = interpolate.interp1d(Y_array,dJ_4dYIntegral,kind='cubic',fill_value='extrapolate')
I_0Interp = interpolate.interp1d(Y_array,I_0Integral,kind='cubic',fill_value='extrapolate')
I_1Interp = interpolate.interp1d(Y_array,I_1Integral,kind='cubic',fill_value='extrapolate')
H_0Interp = interpolate.interp1d(Y_array,H_0Integral,kind='cubic',fill_value='extrapolate')

def J_0(Y):
    conditions = [ Y < YinterpMaxN, Y > YinterpMaxP]
    choices = [0.5*np.sqrt(np.pi)*np.exp(Y), np.sqrt(np.abs(Y))]
    return np.select(conditions,choices,J_0Interp(Y))
def dJ_0dY(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.5*np.sqrt(np.pi)*np.exp(Y), 0.5/np.sqrt(np.abs(Y))]
    return np.select(conditions,choices,dJ_0dYInterp(Y))
def J_1(Y):
    conditions = [ Y < YinterpMaxN, Y > YinterpMaxP]
    choices = [0.5*np.sqrt(mt.pi)*(-Y)*np.exp(Y), np.pi**2/(6*np.sqrt(np.abs(Y)))]
    return np.select(conditions,choices,J_1Interp(Y))
def dJ_1dY(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.5*np.sqrt(mt.pi)*(-Y)*np.exp(Y), -np.pi**2/(12*np.abs(Y)**1.5)]
    return np.select(conditions,choices,dJ_1dYInterp(Y))
def J_2(Y):
    conditions = [ Y < YinterpMaxN, Y > YinterpMaxP]
    choices = [0.5*np.sqrt(mt.pi)*(-Y)**2*np.exp(Y), np.pi**2/3*np.sqrt(np.abs(Y))]
    return np.select(conditions,choices,J_2Interp(Y))
def dJ_2dY(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.5*np.sqrt(mt.pi)*(-Y)**2*np.exp(Y), np.pi**2/(6*np.sqrt(np.abs(Y)))]
    return np.select(conditions,choices,dJ_2dYInterp(Y))
def dJ_3dY(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.5*np.sqrt(mt.pi)*(-Y)**3*np.exp(Y), -7*np.pi**4/(60*np.abs(Y)**1.5)]
    return np.select(conditions,choices,dJ_3dYInterp(Y))
def dJ_4dY(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.5*np.sqrt(mt.pi)*(-Y)**4*np.exp(Y), 7*np.pi**4/(30*np.sqrt(np.abs(Y)))]
    return np.select(conditions,choices,dJ_4dYInterp(Y))
def I_0(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.75*np.sqrt(mt.pi)*np.exp(Y), np.abs(Y)**1.5]
    return np.select(conditions,choices,I_0Interp(Y))
def I_1(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [0.75*np.sqrt(mt.pi)*(-Y)*np.exp(Y), 0.5*np.pi**2*np.sqrt(np.abs(Y))]
    return np.select(conditions,choices,I_1Interp(Y))
def H_0(Y):
    conditions = [ Y <= YinterpMaxN, Y >= YinterpMaxP]
    choices = [15/8*np.sqrt(mt.pi)*np.exp(Y), np.abs(Y)**2.5]
    return np.select(conditions,choices,H_0Interp(Y))

#Create interpolating functions for Int1, ..., Int7 used for calculating u_B

a_arrayP1 = np.linspace(-300.5,-50.1,num=50,endpoint=True)
a_arrayP2 = np.linspace(-50,-5.1,num=50,endpoint=True)
a_arrayP3 = np.linspace(-5,0,num=50,endpoint=True)
a_arrayP4 = np.linspace(0.001,20.5,num=50,endpoint=True)
a_array = np.concatenate((a_arrayP1,a_arrayP2,a_arrayP3,a_arrayP4))

Int1_integral = []
Int2_integral = []
Int3_integral = []
Int4_integral = []
Int5_integral = []
Int6_integral = []
Int7_integral = []

def Int1_integrand(x,a):
    if x**2+a > 2e2:
        return np.exp(-(x**2+a))
    else:
        return 1/(np.exp(x**2+a)+1)
def Int2_integrand(x,a):
    if x**2+a > 2e2:
        return x**2*np.exp(-(x**2+a))
    else:
        return x**2/(np.exp(x**2+a)+1)
def Int3_integrand(x,a):
    if x**2+a > 2e2:
        return x**4*np.exp(-(x**2+a))
    else:
        return x**4/(np.exp(x**2+a)+1)
def Int4_integrand(x,a):
    if x**2+a > 2e2:
        return x**6*np.exp(-(x**2+a))
    else:
        return x**6/(np.exp(x**2+a)+1)
def Int5_integrand(x,a):
    if x**2+a > 2e2:
        return x**8*np.exp(-(x**2+a))
    else:
        return x**8/(np.exp(x**2+a)+1)
def Int6_integrand(x,a):
    if x**2+a > 2e2:
        return x**10*np.exp(-(x**2+a))
    else:
        return x**10/(np.exp(x**2+a)+1)
def Int7_integrand(x,a):
    if x**2+a > 2e2:
        return x**12*np.exp(-(x**2+a))
    else:
        return x**12/(np.exp(x**2+a)+1)

for a in a_array:
    Int1_integral.append(integrate.quad(Int1_integrand, 0, np.inf, args=(a))[0])
    Int2_integral.append(integrate.quad(Int2_integrand, 0, np.inf, args=(a))[0])
    Int3_integral.append(integrate.quad(Int3_integrand, 0, np.inf, args=(a))[0])
    Int4_integral.append(integrate.quad(Int4_integrand, 0, np.inf, args=(a))[0])
    Int5_integral.append(integrate.quad(Int5_integrand, 0, np.inf, args=(a))[0])
    Int6_integral.append(integrate.quad(Int6_integrand, 0, np.inf, args=(a))[0])
    Int7_integral.append(integrate.quad(Int7_integrand, 0, np.inf, args=(a))[0])

Int1interp = interpolate.interp1d(a_array,Int1_integral,kind='cubic',fill_value='extrapolate')
Int2interp = interpolate.interp1d(a_array,Int2_integral,kind='cubic',fill_value='extrapolate')
Int3interp = interpolate.interp1d(a_array,Int3_integral,kind='cubic',fill_value='extrapolate')
Int4interp = interpolate.interp1d(a_array,Int4_integral,kind='cubic',fill_value='extrapolate')
Int5interp = interpolate.interp1d(a_array,Int5_integral,kind='cubic',fill_value='extrapolate')
Int6interp = interpolate.interp1d(a_array,Int6_integral,kind='cubic',fill_value='extrapolate')
Int7interp = interpolate.interp1d(a_array,Int7_integral,kind='cubic',fill_value='extrapolate')

#Loads interpolating functions h1 and h2 for high T/high n_max calculations of thermodynamic functions (generated using "uBBTempDep.py")
with open('h1.pkl', 'rb') as f:
    h1Interp = pickle.load(f)
with open('h2.pkl', 'rb') as f:
    h2Interp = pickle.load(f)    
#Loads interpolating function for ln( integral_1^infinity dx*sqrt(x^2-1)/(exp(a*x-b)+1) ). Arguments are a and b
with open('LogFOm.pkl', 'rb') as f:
    LogFOmInterp = pickle.load(f)
#Loads interpolating function for ln(G_1(a,b)) = ln( integral_1^infinity dx*x^2/sqrt(x^2-1)/(exp(a*x-b)+1) ). Arguments are a and b
with open('LogG1.pkl', 'rb') as f:
    LogG1Interp = pickle.load(f)
#Loads interpolating function for ln(G_2(a,b)) = ln( integral_1^infinity dx*1/sqrt(x^2-1)/(exp(a*x-b)+1) ). Arguments are a and b
with open('LogG2.pkl', 'rb') as f:
    LogG2Interp = pickle.load(f)
#Loads interpolating function for ln(-H_2(a,b)) = ln( a*integral_1^infinity dx*x/sqrt(x^2-1)*exp(a*x-b)/(exp(a*x-b)+1)^2 ). Arguments are a and b
with open('LogNH2.pkl', 'rb') as f:
    LogNH2Interp = pickle.load(f)
#Loads interpolating function for ln(I_1(a,b)) = ln( a*integral_1^infinity dx*x/sqrt(x^2-1)/(exp(a*x-b)+1) ). Arguments are a and b
with open('LogI1.pkl', 'rb') as f:
    LogI1Interp = pickle.load(f)      
#Loads interpolating function for ln(-I_2(a,b)) = ln( a*integral_1^infinity dx*1/sqrt(x^2-1)*exp(a*x-b)/(exp(a*x-b)+1)^2 ). Arguments are a and b
with open('LogNI2.pkl', 'rb') as f:
    LogNI2Interp = pickle.load(f)          
    
#Loads interpolating functions for polylogarithms of order s = -1/2, 1/2, 3/2, 5/2, 7/2 and 9/2. Argument is z where function being evaluated is Li_s(-e^z)
with open('PLN1_2.pkl', 'rb') as f:
    PolyLogN1_2Interp = pickle.load(f)
with open('PL1_2.pkl', 'rb') as f:
    PolyLog1_2Interp = pickle.load(f)
with open('PL3_2.pkl', 'rb') as f:
    PolyLog3_2Interp = pickle.load(f)
with open('PL5_2.pkl', 'rb') as f:
    PolyLog5_2Interp = pickle.load(f)   
with open('PL7_2.pkl', 'rb') as f:
    PolyLog7_2Interp = pickle.load(f)
with open('PL9_2.pkl', 'rb') as f:
    PolyLog9_2Interp = pickle.load(f)

#Other interpolating functions used in the high T/high n_max calculations of thermodynamic functions
X_array = np.logspace(-3,4,200)

def i1Integrand(y,X):
    return 1/y**2.5*(y/np.tanh(y)-1)*(np.exp(-y*X)-1)
def i2Integrand(y,X):
    return 1/y**1.5*(y/np.tanh(y)-1)*np.exp(-y*X)
def i3Integrand(y,X):
    return 1/y**0.5*(y/np.tanh(y)-1)*np.exp(-y*X)

i1Integral = []
i2Integral = []
i3Integral = []

for X in X_array:
    i1Integral.append(integrate.quad(i1Integrand, 0, np.inf, args=(X), epsrel=5e-9, maxp1=150,full_output=1)[0] + np.sqrt(2/np.pi)*special.zeta(1.5))
    i2Integral.append(integrate.quad(i2Integrand, 0, 3e2/X, args=(X), epsrel=5e-9, maxp1=150)[0])
    i3Integral.append(integrate.quad(i3Integrand, 0, 3e2/X, args=(X), epsrel=5e-9, maxp1=150)[0])

i1Interp = interpolate.interp1d(X_array,i1Integral,kind='cubic',fill_value='extrapolate')
i2Interp = interpolate.interp1d(X_array,i2Integral,kind='cubic',fill_value='extrapolate')
i3Interp = interpolate.interp1d(X_array,i3Integral,kind='cubic',fill_value='extrapolate')

#Function which prints string "string" if root node
def parprint(NodeRank,string):
    if(NodeRank == 0):
        print(string)

#####################################################################################################

# Parameters
LandauQuantThermo = 1 #set to zero to exclude Landau quantization effects from thermodynamics and 1 to include them
InitialB = 0 #set to zero to use a large-scale initial field with only a few Fourier modes, and 1 to use a "flat" initial field spectrum
HTOn = 1 #set to zero to turn off Hall term and 1 to include it
if LandauQuantThermo == 0: EROn = 0 #set eddy resistivity off if not including Landau quantization effects from thermodynamics
T_init = 2e8/T_0 #initial (uniform) temperature in reduced units
B_initMax = 5e14/B_0 #prefactor for initial magnetic field
p_B0 = 1 #number of oscillations of B (large-scale initial field only)
beta_0 = 10 #ratio between z-dependent part of A and y-dependent part (dimensionless, large-scale initial field only)
mu_eConst = 35 #fixed value of mu_e in MeV
nu_0 = 0 #amplitude of n_e oscillation (dimensionless)
p_n0 = 1 #number of oscillations of n_e
Nx, Ny, Nz = 2, 128, 128 #number of grid points in x, y, z directions
Lx, Ly, Lz, = 10, 10, 10 #length of simulation domain in each direction in reduced units
dealias = (1,1.5,1.5) #dealiasing factor in each direction
stop_sim_time = 1e1*yr/tau
timestepper = d3.RK222
dtype = np.float64
#mesh = [1,8] #Core mesh.
mesh = None #Core mesh. Must be compatible with number of cores being used. Use "None" if using a single core
ncores = np.prod(mesh) #Number of cores

parprint(MPI.COMM_WORLD.Get_rank(),f'Run for {stop_sim_time*tau/yr:.1f} years with LQT = {LandauQuantThermo}; Hall term = {HTOn}')

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = d3.RealFourier(coords['x'],size=Nx,bounds=(0,Lx),dealias=dealias[0])
ybasis = d3.RealFourier(coords['y'],size=Ny,bounds=(0,Ly),dealias=dealias[1])
zbasis = d3.RealFourier(coords['z'],size=Nz,bounds=(0,Lz),dealias=dealias[2])

domain = Domain(dist, (xbasis,ybasis,zbasis))
V = Lx*Ly*Lz #volume of simuation domain in reduced units

# Fields
T = dist.Field(name='T', bases=(xbasis,ybasis,zbasis))
T.change_scales(dealias)
A = dist.VectorField(coords, name='A', bases=(xbasis,ybasis,zbasis))
A.change_scales(dealias)
φ = dist.Field(name='φ', bases=(xbasis,ybasis,zbasis))
τ_φ = dist.Field(name='τ_φ')

# Substitutions
x = dist.local_grid(xbasis, scale=dealias[0])
y = dist.local_grid(ybasis, scale=dealias[1])
z = dist.local_grid(zbasis, scale=dealias[2])
ex, ey, ez = coords.unit_vector_fields(dist)
x_array = x[:,0,0] #array of x grid points
y_array = y[0,:,0] #array of y grid points
z_array = z[0,0,:] #array of z grid points

# Coefficients/background structure fields
n_e = dist.Field(name='n_e', bases=(ybasis,zbasis)) #electron number density in fm^{-3}.
n_ebarInv = dist.Field(name='n_ebarInv', bases=(ybasis,zbasis)) #electron number density in reduced units.
mu_e = dist.Field(name='mu_e', bases=(ybasis,zbasis)) #electron chemical potential in MeV.
mu_e2 = dist.Field(name='mu_e2', bases=(xbasis,ybasis,zbasis)) #copy of mu_e; only used to take gradient of in equations of motion
mu_e.change_scales(dealias)
mu_e2.change_scales(dealias)
n_e.change_scales(dealias)
n_ebarInv.change_scales(dealias)

n_ei = dist.Field(name='n_ei', bases=(ybasis,zbasis)) #electron number density in fm^{-3}. 
n_ei.change_scales(dealias)
n_ebarInvi = dist.Field(name='n_ebarInvi', bases=(ybasis,zbasis)) #electron number density in reduced units.
n_ebarInvi.change_scales(dealias)

mu_e['g'] = mu_eConst*( 1 + nu_0*np.cos(2*np.pi*p_n0*z/Lz) ) #electron chemical potential in MeV
mu_e2['g'] = mu_e['g']

Mag = dist.Field(name='Mag', bases=(ybasis,zbasis))
Mag.change_scales(dealias)
Omega_BB = dist.Field(name='Omega_BB', bases=(ybasis,zbasis)) #d^2Omega/dB^2=-chi_mu=-dM/dB at fixed T and mu_e (dimensionless)
Omega_BB.change_scales(dealias)
Omega_Bmu = dist.Field(name='Omega_Bn', bases=(ybasis,zbasis)) #d^2Omega/dBdmu_e=-dM/dmu_e at fixed T (dimensionless)
Omega_Bmu.change_scales(dealias)

Bmag = dist.Field(name='Bmag', bases=(xbasis,ybasis,zbasis)) #magnitude of magnetic field in reduced units
Bmag.change_scales(dealias)
Bhat = dist.VectorField(coords, name='Bhat', bases=(xbasis,ybasis,zbasis)) #Unit vector in direction of magnetic field
Bhat.change_scales(dealias)
D_H = dist.Field(name='D_H', bases=(ybasis,zbasis)) #Hall diffusivity
D_H.change_scales(dealias)
D_Ohm = dist.Field(name='D_Ohm', bases=(ybasis,zbasis)) #Ohmic diffusivity perpendicular to magnetic field
D_Ohm.change_scales(dealias)
D_OhmFluc = dist.Field(name='D_OhmFluc', bases=(ybasis,zbasis)) #fluctuating (difference from spatial mean) part of Ohmic diffusivity
D_OhmFluc.change_scales(dealias)

# Initial conditions
T['g'] = T_init #Uniform initial temperature

A_i = dist.VectorField(coords, name='A_i', bases=(xbasis,ybasis,zbasis))
A_i.change_scales(dealias)

if InitialB == 0:
    #Large-scale field consisting of a few Fourier modes
    A_i['g'][0] = B_initMax*( Ly/(2*np.pi)*np.cos( y/Ly*2*np.pi ) + beta_0*Lz/(2*np.pi*p_B0)*np.sin( z/Lz*2*np.pi*p_B0 ) )
    A_i['g'][1] = B_initMax*( beta_0*Lz/(2*np.pi*p_B0)*np.sin( z/Lz*2*np.pi*p_B0 ) )
    A_i['g'][2] = B_initMax*( Ly/(2*np.pi)*np.sin( y/Ly*2*np.pi ) )
    
elif InitialB == 1:
    
    ky = np.floor(dist.local_modes(ybasis)/2)
    kz = np.floor(dist.local_modes(zbasis)/2)
    
    def SFlat(k,k1,k0):
        return np.where(np.logical_and(k<=k1,k >k0), 1/k, 0)
    
    A_i['c'][0,0] = B_initMax*SFlat(np.sqrt(ky**2+kz**2),10,5)*np.sqrt(3)
    A_i['c'][1,0] = B_initMax*SFlat(np.sqrt(ky**2+kz**2),10,5)*np.sqrt(3)*(1 - ky*ky/(1e-20+ky**2+kz**2)*1 - ky*kz/(1e-20+ky**2+kz**2)*1)
    A_i['c'][2,0] = B_initMax*SFlat(np.sqrt(ky**2+kz**2),10,5)*mp.sqrt(3)*(1 - kz*ky/(1e-20+ky**2+kz**2)*1 - kz*kz/(1e-20+ky**2+kz**2)*1)

else:
    parprint(MPI.COMM_WORLD.Get_rank(),"Invalid value for InitialB; must be either 0 (large-scale initial field) or 1 (turbulent, 'flat' initial field spectrum)")

parprint(MPI.COMM_WORLD.Get_rank(),"A initialized")

#Filter out noise from spectral transform
A['c'] = A_i['c']
A['c'][np.abs(A['c']) < 1e-15] = 0

B_eval = d3.curl(A)
UB0Vec = (d3.integ((B_eval@B_eval),coords)).evaluate()['g']/(8*np.pi)*B_0**2*L_0**3
B_rmsVec = B_0*np.sqrt(1/V*d3.integ((B_eval@B_eval),coords).evaluate()['g'])

if(UB0Vec.size == 0):
    UB0Vec = np.array([0.0])
else:
    UB0Vec = np.array([UB0Vec.item()])

UB0 = np.zeros(1)
MPI.COMM_WORLD.Allreduce(UB0Vec, UB0, op=MPI.SUM)

if(B_rmsVec.size == 0):
    B_rmsVec = np.array([0.0])
else:
    B_rmsVec = np.array([B_rmsVec.item()])
    
B_rms = np.zeros(1)
MPI.COMM_WORLD.Allreduce(B_rmsVec, B_rms, op=MPI.SUM) #note that the initial B_rms value will not be accurate because of Jacobian factors

MPI.COMM_WORLD.Barrier()
Bmag['g'] = np.sqrt(B_eval@B_eval).evaluate()['g']
Bhat['g'] = (B_eval/Bmag).evaluate()['g']

# Interpolation from background stellar model data to determine values for non-evolving coefficients
n_e['g'] = n_eCalc(Bmag,mu_e,M_e,T,LandauQuantThermo)
n_ebarInv['g'] = n_e0/n_e['g'] #1/n_e in reduced units

MPI.COMM_WORLD.Barrier()

if LandauQuantThermo == 1:
    Mag['g'], Omega_BB['g'], Omega_Bmu['g'] = Omega_xyFixedT(Bmag, mu_e, M_e, T)
    parprint(MPI.COMM_WORLD.Get_rank(),"f_BB -- f_Bn initialized")
        
#Articially enhanced Ohmic diffusivity
ElecConductivityConst = 5e22
D_Ohm['g'] = c**2*tau/(4*np.pi*L_0**2)/ElecConductivityConst
D_OhmFunc = dist.Field(name='D_OhmFunc')
D_OhmBarFactor = 1000
D_OhmConst = D_OhmBarFactor*c**2*tau/(4*np.pi*L_0**2)/ElecConductivityConst
D_OhmFunc['g'] = D_OhmConst
D_OhmFluc['g'] = D_Ohm['g'] - D_OhmConst

timestep = 1e4/tau #timestep in reduced units

#Problem
MPI.COMM_WORLD.Barrier()
problem = d3.IVP([A,φ,τ_φ], namespace=locals())

#Terms with non-constant coefficients must be placed on right-hand side of equation (i.e., not side with time derivative)
B = d3.curl(A)
B_mag = np.sqrt(B@B)
H = d3.curl(A)*(1-4*np.pi*Mag/B_mag)
B_hat = B/Bmag
gradmu = d3.grad(mu_e2) #gradient of mu_e in MeV/reduced length
gradB = d3.grad(Bmag)
Jpar = -Bhat*(Bhat@d3.lap(A)) #current density parallel to magnetic field times 4*pi/c
D_H = HTOn*c*tau*B_0*n_ebarInv/(4*np.pi*unit_e*L_0**2*n_e0*1e39) #evolves as n_e evolves

#Can only include linear terms with constant coefficients on left-hand side; linear terms with non-constant coefficients are split into constant*linear and fluctuating*linear terms, with the latter on the right-hand side
if LandauQuantThermo == 0:
    J = -d3.lap(A) #current density times 4*pi/c in Coulomb gauge
    problem.add_equation("dt(A) - D_OhmFunc*lap(A) + grad(φ) = D_OhmFluc*lap(A) - D_H*cross(J,B)")
elif LandauQuantThermo == 1:
    Jgrad = 4*np.pi*Mag/B_mag*d3.lap(A) + 4*np.pi*( (Omega_BB/B_mag+Mag/B_mag**2)*d3.cross(gradB,B) + Omega_Bmu/B_mag*d3.cross(gradmu,B) ) #gradient parts of current density times 4*pi/c in Coulomb gauge
    J = -d3.lap(A) + Jgrad #current density times 4*pi/c in Coulomb gauge
    problem.add_equation("dt(A) - D_OhmFunc*lap(A) + grad(φ) = D_OhmFluc*lap(A) - D_Ohm*Jgrad - D_H*cross(J,B)")
else:
    parprint(MPI.COMM_WORLD.Get_rank(),"Invalid value for LandauQuantThermo; must be either 0 (not included) or 1 (included)")

problem.add_equation("div(A) + τ_φ = 0") #Coulomb gauge condition
problem.add_equation("integ(φ) = 0") #Electric potential gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
J_B = -d3.lap(A) #current density for B=H times 4*pi/c in Coulomb gauge
E_perp = D_Ohm*J + D_H*d3.cross(J,B) #E field perpendicular to B in reduced units. When ExB is performed, component of J parallel to B is zero
u_B = B@B/(8*np.pi)*B_0**2*L_0**3
E_BIntegrated = d3.integ(u_B,coords) #volume-integrated magnetic field energy (erg)

if LandauQuantThermo == 1:
    QuasiJH_density = B_0**2*L_0**3/(4*np.pi*tau)*( ( D_Ohm*J + D_H*d3.cross(J,B) )@J_B ) #Joule heating rate (erg/s/reduced volume)
    JH_density = B_0**2*L_0**3/(4*np.pi*tau)*( ( D_Ohm*J )@J )
else:
    QuasiJH_density = B_0**2*L_0**3/(4*np.pi*tau)*( D_Ohm*J@J_B ) #Quasi-Joule heating rate (erg/s/reduced volume)
    JH_density = B_0**2*L_0**3/(4*np.pi*tau)*( D_Ohm*J@J ) #Joule heating rate (erg/s/reduced volume)

QuasiJouleHeating = d3.integ(QuasiJH_density,coords) #volume-integrated Joule heating (erg/s)
JouleHeating = d3.integ(JH_density,coords) #volume-integrated Joule heating (erg/s)

outputFolder = 'MagEvoParFouFM_snapshots'
snapshots = solver.evaluator.add_file_handler(outputFolder, sim_dt=stop_sim_time/1000, max_writes=1000)
snapshots.add_task(E_BIntegrated, name='E_B') #integrated magnetic field energy in erg
snapshots.add_task(QuasiJouleHeating, name='QuasiJH') #integrated volumetric quasi-Joule heating rate in erg/s
snapshots.add_task(JouleHeating, name='JH') #integrated volumetric Joule heating rate in erg/s
snapshots.add_task(B, scales=dealias, name='B_full') #magnetic field in position space in reduced units
snapshots.add_task(B, layout='c', name='B_full_c') #magnetic field in spectral (Fourier) space in reduced units

report_cadence = 10 #cadence for computing global properties
flow = d3.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(u_B, name='MagneticEnergy_density')
flow.add_property(QuasiJH_density, name='QuasiJouleHeating_density')
flow.add_property(JH_density, name='JouleHeating_density')
flow.add_property(B@B, name='BdotB')

QuasiJHVec = QuasiJouleHeating.evaluate()['g']
JHVec = JouleHeating.evaluate()['g']

if(QuasiJHVec.size == 0):
    QuasiJHVec = np.array([0.0])
    JHVec = np.array([0.0])
else:
    QuasiJHVec = np.array([QuasiJHVec.item()])
    JHVec = np.array([JHVec.item()])

QuasiJH_0 = np.zeros(1)
MPI.COMM_WORLD.Allreduce(QuasiJHVec, QuasiJH_0, op=MPI.SUM)

JH_0 = np.zeros(1)
MPI.COMM_WORLD.Allreduce(JHVec, JH_0, op=MPI.SUM)

tsteps = [0]
QuasiJH_list = [QuasiJH_0]
JH_list = [JH_0]

addattrs = 0
DeltaUBprev = 0 #initializes value of DeltaUB (change in magnetic field energy from initial value) from previous time step

MPI.COMM_WORLD.Barrier()

#Main loop
try:
    logger.info('Starting main loop')
    logger.info('It.=%i, t=%.2e s, dt=%.1e s, B_rms=%.6e G, ΔU_B=0, JH = 0' %(solver.iteration, solver.sim_time*tau, timestep*tau, B_rms))
    
    while solver.proceed:
        solver.step(timestep)
        
        #Update non-constant coefficients using new values of B/T
        Bmag['g'] = B_mag.evaluate()['g']
        Bhat['g'] = B_hat.evaluate()['g']
       
        #If using Landau quantization in thermodynamics, update coefficients
        if LandauQuantThermo == 1:
            mu_e['g']
            n_e['g'] = n_eCalc(Bmag, mu_e, M_e, T, LandauQuantThermo)
            n_ebarInv['g'] = n_e0/n_e['g'] #1/n_e in reduced units
            Mag['g'], Omega_BB['g'], Omega_Bmu['g'] = Omega_xyFixedT(Bmag, mu_e, M_e, T)

        if (solver.iteration-1) % report_cadence == 0:
            B_rms = B_0*np.sqrt(1/V*flow.volume_integral('BdotB'))
            u_B.evaluate() #need to do this for flow.volume_integral('MagneticEnergy_density') to give the correct answer
            QuasiJH_density.evaluate() #need to do this for flow.volume_integral('QuasiJouleHeating_density') to give the correct answer
            DeltaUB = flow.volume_integral('MagneticEnergy_density')-UB0
            tsteps = np.append(tsteps, solver.sim_time*tau)
            QuasiJH_list = np.append(QuasiJH_list, flow.volume_integral('QuasiJouleHeating_density'))
            QuasiJH = np.trapz( QuasiJH_list, tsteps )
            FracDiff = ( DeltaUB + QuasiJH )/abs(QuasiJH) #fractional difference for energy conservation
            logger.info('It.=%i, t=%.2e s, dt=%.1e s, B_rms=%.6e G, ΔU_B=%.6e erg, QuasiJH=%.6e erg, E.Con.=%.4e' %(solver.iteration, solver.sim_time*tau, timestep*tau, B_rms, DeltaUB, QuasiJH, FracDiff))

        if (DeltaUBprev < DeltaUB):
            parprint(MPI.COMM_WORLD.Get_rank(),'Probable numerical instability: U_B growing')
        DeltaUBprev = DeltaUB
            
        if(MPI.COMM_WORLD.Get_rank() == 0 and addattrs==0): #Only do this in master process
            with h5py.File(outputFolder+'/'+outputFolder+'_s1.h5', mode='r+') as file:

                file.attrs.create('number_gridpoints_x',Nx)
                file.attrs.create('number_gridpoints_y',Ny)
                file.attrs.create('number_gridpoints_z',Nz)
                file.attrs.create('tau',tau)
                file.attrs.create('B_0',B_0)
                file.attrs.create('L_0',L_0)
                file.attrs.create('T_0',T_0)
                file.attrs.create('T_init',T_init)
                addattrs = 1
     
    #Print final iteration information
    MPI.COMM_WORLD.Barrier()
    B_rms = B_0*np.sqrt(1/V*flow.volume_integral('BdotB'))
    u_B.evaluate() #need to do this for flow.volume_integral('MagneticEnergy_density') to give the correct answer
    QuasiJH_density.evaluate() #need to do this for flow.volume_integral('QuasiJouleHeating_density') to give the correct answer
    DeltaUB = flow.volume_integral('MagneticEnergy_density')-UB0
    tsteps = np.append(tsteps,solver.sim_time*tau)
    QuasiJH_list = np.append(QuasiJH_list, flow.volume_integral('QuasiJouleHeating_density'))

    QuasiJH = np.trapz( QuasiJH_list, tsteps )
    FracDiff = ( DeltaUB + QuasiJH )/abs(QuasiJH) #fractional difference for energy conservation
    logger.info('It.=%i, t=%.2e s, dt=%.1e s, B_rms=%.6e G, ΔU_B=%.6e erg, QuasiJH=%.6e erg, E.Con.=%.4e' %(solver.iteration, solver.sim_time*tau, timestep*tau, B_rms, DeltaUB, QuasiJH, FracDiff))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()