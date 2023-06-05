import math
import numpy as np
M_PI = math.pi

Mc = 2

# Note that Ni-d and O-p orbitals use hole language
# ed = {'d3z2r2': 0.0,\
#       'dx2y2' : 0.0,\
#       'dxy'   : 0.0,\
#       'dxz'   : 0.0,\
#       'dyz'   : 0.0}

edCu = {'d3z2r2': 1.97,\
        'dx2y2' : 0.0,\
        'dxy'   : 1.53,\
        'dxz'   : 1.6,\
        'dyz'   : 1.6}

# after adding H
edCu = {'d3z2r2': 0.35,\
        'dx2y2' : 0.0,\
        'dxy'   : 1.55,\
        'dxz'   : 1.9,\
        'dyz'   : 1.9}

edNi = edCu

epNis = np.arange(4.7, 4.71, 1.0)
epCus = epNis

ess = np.arange(4.8, 4.81, 1.0)

ANis = np.arange(6.0, 6.01, 1.0)
ACus = ANis

B = 0.15
C = 0.58
#As = np.arange(100, 100.1, 1.0)
# ANis = np.arange(0.0, 0.01, 1.0)
# ACus = np.arange(0.0, 0.01, 1.0)
# B = 0
# C = 0

# Note: tpd and tpp are only amplitude signs are considered separately in hamiltonian.py
# Slater Koster integrals and the overlaps between px and d_x^2-y^2 is sqrt(3) bigger than between px and d_3z^2-r^2 
# These two are proportional to tpd,sigma, whereas the hopping between py and d_xy is proportional to tpd,pi

# IMPORTANT: keep all hoppings below positive to avoid confusion
#            hopping signs are considered in dispersion separately
Norb = 5
if Norb==8 or Norb==5:
    #tpds = [0.00001]  # for check_CuO4_eigenvalues.py
    tpds = np.linspace(1.3, 1.3, num=1, endpoint=True) #[0.25]
#     tpds = [0.01]
    tpps = [0.5]
elif Norb==9 or Norb==11:    
    # pdp = sqrt(3)/4*pds so that tpd(b2)=tpd(b1)/2: see Eskes's thesis and 1990 paper
    # the values of pds and pdp between papers have factor of 2 difference
    # here use Eskes's thesis Page 4
    # also note that tpd ~ pds*sqrt(3)/2
    vals = np.linspace(1.3, 1.3, num=1, endpoint=True)
    pdss = np.asarray(vals)*2./np.sqrt(3)
    pdps = np.asarray(pdss)*np.sqrt(3)/4.
#     pdss = [0.01]
#     pdps = [0.01]
    #------------------------------------------------------------------------------
    # note that tpp ~ (pps+ppp)/2
    # because 3 or 7 orbital bandwidth is 8*tpp while 9 orbital has 4*(pps+ppp)
    pps = 0.9
    ppp = 0.2
         
#     pps = 0.01
#     ppp = 0.01

tzb1s = np.arange(0.3, 0.31, 1.0)  
tza1s = np.arange(0.36, 0.37, 1.0)  

tdss = [1.63]
#tdss = np.linspace(3.6, 3.7, num=11, endpoint=True)
tpss = [0.58]
    
if_tz_exist = 2
    #if if_tz_exist = 0,tz exist in all orbits.
    #if if_tz_exist = 1,tz exist in d orbits.
    #if if_tz_exist = 2,tz exist in d3z2r2 orbits.    
    
wmin = -10; wmax = 30
eta = 0.1
Lanczos_maxiter = 600

# restriction on variational space
reduce_VS = 1

if_H0_rotate_byU = 1
basis_change_type = 'd_double' # 'all_states' or 'd_double'
if_print_VS_after_basis_change = 0

if_compute_Aw = 0
if if_compute_Aw==1:
    if_find_lowpeak = 0
    if if_find_lowpeak==1:
        peak_mode = 'lowest_peak' # 'lowest_peak' or 'highest_peak' or 'lowest_peak_intensity'
        if_write_lowpeak_ep_tpd = 1
    if_write_Aw = 0
    if_savefig_Aw = 1

if_get_ground_state = 1
if if_get_ground_state==1:
    # see issue https://github.com/scipy/scipy/issues/5612
    Neval = 10
if_compute_Aw_dd_total = 0

if Norb==8 or Norb==10 or Norb==12:
    Ni_Cu_orbs = ['dx2y2','dxy','dxz','dyz','d3z2r2']
elif Norb==5:
    Ni_Cu_orbs = ['dx2y2','d3z2r2']    

H_orbs= ['s']


    
if Norb==8 or Norb==5:
    O1_orbs  = ['px']
    O2_orbs  = ['py']
elif Norb==10:
    O1_orbs  = ['px1','py1']
    O2_orbs  = ['px2','py2']
elif Norb==12:
    O1_orbs  = ['px1','py1','pz1']
    O2_orbs  = ['px2','py2','pz2']
O_orbs = O1_orbs + O2_orbs
# sort the list to facilliate the setup of interaction matrix elements
Ni_Cu_orbs.sort()
H_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
O_orbs.sort()
print ("Ni_Cu_orbs = ", Ni_Cu_orbs)
print ("H_orbs = ", H_orbs)
print ("O1_orbs = ",  O1_orbs)
print ("O2_orbs = ",  O2_orbs)
orbs = Ni_Cu_orbs + H_orbs + O_orbs 
#assert(len(orbs)==Norb)

Upps = [0]
Usss = [0]
symmetries = ['1A1','3B1','3B1','1A2','3A2','1E','3E']
print ("compute A(w) for symmetries = ",symmetries)

