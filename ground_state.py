import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util
import lanczos

def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    
    if orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        elif z2==z1 and orb1=='dx2y2' and orb2=='d3z2r2':
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
    elif orb1 in pam.O_orbs and orb2 in pam.Obilayer_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]            
    elif orb1 in pam.Obilayer_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]             
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];    
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
    tlabel = slabel[10:15] + [s4,orb4,x4,y4,z4]
    tmp34 = reorder_z(tlabel)
    
    if tmp34 == tlabel:
        slabel2 = slabel + [s4,orb4,x4,y4,z4]
    elif  tmp34 != tlabel:
        tlabel = slabel[5:10] + [s4,orb4,x4,y4,z4]
        tmp24 = reorder_z(tlabel)
        if tmp24 == tlabel:
            slabel2 = slabel[0:10]+ [s4,orb4,x4,y4,z4] + slabel[10:15]
        elif  tmp24 != tlabel:
            tlabel = slabel[0:5] + [s4,orb4,x4,y4,z4]   
            tmp14 = reorder_z(tlabel)
            if tmp14 == tlabel:
                slabel2 = slabel[0:5]+ [s4,orb4,x4,y4,z4] + slabel[5:15]
            elif  tmp14 != tlabel:
                slabel2 = [s4,orb4,x4,y4,z4] + slabel[0:15]     
                
    return slabel2


def get_ground_state(matrix, VS, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val,bonding_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     print ('H=')
#     print (M_dense)
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()                                                               #calculate atom limit
#     print ('lowest eigenvalue of H from np.linalg.eigh = ')
#     print (vals)
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    print (vals[0])
    
    if abs(vals[0]-vals[3])<10**(-5):
        number = 4
    elif abs(vals[0]-vals[2])<10**(-5):
        number = 3        
    elif abs(vals[0]-vals[1])<10**(-5):
        number = 2
    else:
        number = 1
    print ('Degeneracy of ground state is ' ,number)        

    
    wgt_LmLn = np.zeros(50)
    wgt_d9Ld10L2 = np.zeros(50)
    wgt_d9d10L3 = np.zeros(50)
    wgt_d10L3d9 = np.zeros(50)        
    wgt_d9L2d10L= np.zeros(50)
    wgt_d10Ld9L2= np.zeros(50)
    wgt_d10d9L3= np.zeros(50)   
    wgt_d10L2d9L= np.zeros(50)  
    wgt_d8Ld10L= np.zeros(50)
    wgt_d10Ld8L= np.zeros(50)        
    wgt_d8d10L2= np.zeros(50)
    wgt_d10d8L2 = np.zeros(50)
    wgt_d8L2d10 = np.zeros(50)        
    wgt_d10L2d8 = np.zeros(50) 
    wgt_d9L2d9 = np.zeros(50)
    wgt_d9d9L2 = np.zeros(50)    
    wgt_d9Ld9L = np.zeros(50)         
    wgt_d9d8L = np.zeros(50)     
    wgt_d8d9L = np.zeros(50) 
    wgt_d9Ld8 = np.zeros(50)  
    wgt_d8Ld9 = np.zeros(50)
    wgt_d8d8 = np.zeros(50)     
    wgt_H = np.zeros(50) 
    wgt_s = np.zeros(50) 
    wgt_ddss = np.zeros(50)    
    wgt_ddsp = np.zeros(50)           
    wgt_dssp = np.zeros(50)         
    wgt_dspp = np.zeros(50)    
    wgt_ddds = np.zeros(50)               

    wgt_a1= np.zeros(10)   
    wgt_b1= np.zeros(10)       
    wgt_L= np.zeros(10) 
    wgt_O= np.zeros(10)     
    text= np.zeros(10)       
    
    #get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,number):                                                                          #gai
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.1)


      
    
#         s11=0
#         s10=0        
#         s01=0
#         s00=0        
        #Sumweight refers to the general weight.Sumweight1 refers to the weight in indices.Sumweight_picture refers to the weight that is calculated.Sumweight2 refers to the weight that differs by orbits

        sumweight=0
        sumweight1=0
        synweight2=0
        # stores all weights for sorting later
        dim = len(vecs[:,k])
        allwgts = np.zeros(dim)
        allwgts = abs(vecs[:,k])**2
        ilead = np.argsort(-allwgts)   # argsort returns small value first by default
            

        total = 0

        print ("Compute the weights in GS (lowest Aw peak)")
        
        #for i in indices[0]:
        for i in range(dim):
            # state is original state but its orbital info remains after basis change
            istate = ilead[i]
            weight = allwgts[istate]
            
            #if weight>0.01:

            total += weight
                
            state = VS.get_state(VS.lookup_tbl[istate])

        
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            s4 = state['hole4_spin']            
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb']            
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']            

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S_Ni_12  = S_Ni_val[istate]
            Sz_Ni_12 = Sz_Ni_val[istate]
            S_Cu_12  = S_Cu_val[istate]
            Sz_Cu_12 = Sz_Cu_val[istate]
            bonding = bonding_val[istate]            
#             S_Niother_12  = S_other_Ni_val[i]
#             Sz_Niother_12 = Sz_other_Ni_val[i]
#             S_Cuother_12  = S_other_Cu_val[i]
#             Sz_Cuother_12 = Sz_other_Cu_val[i]
             
#             print (' state ', i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,\
#                '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
#                ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
#                ", weight = ", weight,'\n')   
                    
    
    
    
            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
            s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];     
            
            
            
            if weight > 0.01:
                sumweight1=sumweight1+abs(vecs[i,k])**2
                print (' state ', istate, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,\
                   '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
                   ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
                   ", weight = ", weight,'\n')   


            if bonding==0:
                text[0]+=abs(vecs[istate,k])**2 
            if bonding==1:
                text[1]+=abs(vecs[istate,k])**2 
            if bonding==-1:
                text[2]+=abs(vecs[istate,k])**2                 
                
                
            if orb1 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb1 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb1 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb1 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2       
                
            if orb2 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb2 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb2 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb2 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2                       
                
            if orb3 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb3 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb3 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb3 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2       
                
            if orb4 =='d3z2r2': 
                wgt_a1[0]+=abs(vecs[istate,k])**2      
            elif orb4 =='dx2y2':    
                wgt_b1[0]+=abs(vecs[istate,k])**2   
            elif orb4 in pam.O_orbs:    
                wgt_L[0]+=abs(vecs[istate,k])**2  
            elif orb4 in pam.Obilayer_orbs:    
                wgt_O[0]+=abs(vecs[istate,k])**2                  
                
                
                
            if (orb1 in pam.O_orbs) and  (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)  and  (orb4 in pam.O_orbs):
                wgt_LmLn[0]+=abs(vecs[istate,k])**2 
                if S_Ni_12==0 and S_Cu_12==0:                     
                    wgt_LmLn[1]+=abs(vecs[istate,k])**2     
                elif S_Ni_12==1 and S_Cu_12==0:                     
                    wgt_LmLn[2]+=abs(vecs[istate,k])**2                         
                elif S_Ni_12==0 and S_Cu_12==1:                     
                    wgt_LmLn[3]+=abs(vecs[istate,k])**2                         
                elif S_Ni_12==1 and S_Cu_12==1:                     
                    wgt_LmLn[4]+=abs(vecs[istate,k])**2   
                if bonding==0:
                    wgt_LmLn[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_LmLn[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_LmLn[48]+=abs(vecs[istate,k])**2                       
                    
                    

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)\
                      and (orb4 in pam.O_orbs)  and z1==z2==2 and z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d9Ld10L2[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2': 
                    wgt_d9Ld10L2[1]+=abs(vecs[istate,k])**2   
                if orb1=='dx2y2':
                    if S_Ni_12==0 and S_Cu_12==0:                     
                        wgt_d9Ld10L2[2]+=abs(vecs[istate,k])**2     
                    elif S_Ni_12==1 and S_Cu_12==0:                     
                        wgt_d9Ld10L2[3]+=abs(vecs[istate,k])**2                         
                    elif S_Ni_12==0 and S_Cu_12==1:                     
                        wgt_d9Ld10L2[4]+=abs(vecs[istate,k])**2                         
                    elif S_Ni_12==1 and S_Cu_12==1:                     
                        wgt_d9Ld10L2[5]+=abs(vecs[istate,k])**2                         
                if orb1=='d3z2r2':
                    if S_Ni_12==0 and S_Cu_12==0:                     
                        wgt_d9Ld10L2[6]+=abs(vecs[istate,k])**2     
                    elif S_Ni_12==1 and S_Cu_12==0:                     
                        wgt_d9Ld10L2[7]+=abs(vecs[istate,k])**2                         
                    elif S_Ni_12==0 and S_Cu_12==1:                     
                        wgt_d9Ld10L2[8]+=abs(vecs[istate,k])**2                         
                    elif S_Ni_12==1 and S_Cu_12==1:                     
                        wgt_d9Ld10L2[9]+=abs(vecs[istate,k])**2
                if s1=='up':        
                    wgt_d9Ld10L2[10]+=abs(vecs[istate,k])**2      
                if s1=='dn':        
                    wgt_d9Ld10L2[11]+=abs(vecs[istate,k])**2  
                if bonding==0:
                    wgt_d9Ld10L2[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9Ld10L2[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9Ld10L2[48]+=abs(vecs[istate,k])**2                     
                    
                        
                        
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)\
                      and (orb4 in pam.O_orbs) and z1==2 and z2==z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d9d10L3[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d9d10L3[1]+=abs(vecs[istate,k])**2  
                if bonding==0:
                    wgt_d9d10L3[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9d10L3[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9d10L3[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs)\
                      and (orb4 in pam.O_orbs) and z1==0 and z2==z3==z4==2:
                if orb1=='dx2y2':
                    wgt_d10L3d9[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2': 
                    wgt_d10L3d9[1]+=abs(vecs[istate,k])**2   
                if orb1=='dx2y2':
                    if s1=='up':                     
                        wgt_d10L3d9[2]+=abs(vecs[istate,k])**2     
                    elif s1=='dn':                     
                        wgt_d10L3d9[3]+=abs(vecs[istate,k])**2                                            
                if orb1=='d3z2r2':
                    if s1=='up':                     
                        wgt_d10L3d9[4]+=abs(vecs[istate,k])**2     
                    elif s1=='dn':                     
                        wgt_d10L3d9[5]+=abs(vecs[istate,k])**2     
                if bonding==0:
                    wgt_d10L3d9[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10L3d9[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10L3d9[48]+=abs(vecs[istate,k])**2                         
                        
                        
                        
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==z3==2 and z4==0:
                if orb1=='dx2y2':
                    wgt_d9L2d10L[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d9L2d10L[1]+=abs(vecs[istate,k])**2   
                if bonding==0:
                    wgt_d9L2d10L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L2d10L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L2d10L[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z2==2 and z1==z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d10Ld9L2[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d10Ld9L2[1]+=abs(vecs[istate,k])**2 
                if bonding==0:
                    wgt_d10Ld9L2[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10Ld9L2[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10Ld9L2[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==z3==z4==0:
                if orb1=='dx2y2':
                    wgt_d10d9L3[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d10d9L3[1]+=abs(vecs[istate,k])**2    
                if bonding==0:
                    wgt_d10d9L3[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10d9L3[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10d9L3[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z2==z3==2 and z1==z4==0:
                if orb1=='dx2y2':
                    wgt_d10L2d9L[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2':
                    wgt_d10L2d9L[1]+=abs(vecs[istate,k])**2 
                if bonding==0:
                    wgt_d10L2d9L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10L2d9L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10L2d9L[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==2 and z3==2 and z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==0:
                    wgt_d8Ld10L[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==1:
                    wgt_d8Ld10L[1]+=abs(vecs[istate,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==0:
                    wgt_d8Ld10L[2]+=abs(vecs[istate,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==0:
                    wgt_d8Ld10L[3]+=abs(vecs[istate,k])**2  
                if bonding==0:
                    wgt_d8Ld10L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8Ld10L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8Ld10L[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==0 and z3==2 and z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Cu_12==0:
                    wgt_d10Ld8L[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Cu_12==1:
                    wgt_d10Ld8L[1]+=abs(vecs[istate,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Cu_12==0:
                    wgt_d10Ld8L[2]+=abs(vecs[istate,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Cu_12==0:
                    wgt_d10Ld8L[3]+=abs(vecs[istate,k])**2    
                if bonding==0:
                    wgt_d10Ld8L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10Ld8L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10Ld8L[48]+=abs(vecs[istate,k])**2                     
                    
                    
                                          
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==2 and z3==z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==0:
                    wgt_d8d10L2[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==1:
                    wgt_d8d10L2[1]+=abs(vecs[istate,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==0:
                    wgt_d8d10L2[2]+=abs(vecs[istate,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==0:
                    wgt_d8d10L2[3]+=abs(vecs[istate,k])**2     
                if bonding==0:
                    wgt_d8d10L2[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8d10L2[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8d10L2[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==0 and z3==z4==0 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Cu_12==0:
                    wgt_d10d8L2[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Cu_12==1:
                    wgt_d10d8L2[1]+=abs(vecs[istate,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Cu_12==0:
                    wgt_d10d8L2[2]+=abs(vecs[istate,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Cu_12==0:
                    wgt_d10d8L2[3]+=abs(vecs[istate,k])**2    
                if bonding==0:
                    wgt_d10d8L2[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10d8L2[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10d8L2[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                    
                                     
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==2 and z3==z4==2 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==0:
                    wgt_d8L2d10[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==1:
                    wgt_d8L2d10[1]+=abs(vecs[istate,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==0:
                    wgt_d8L2d10[2]+=abs(vecs[istate,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==0:
                    wgt_d8L2d10[3]+=abs(vecs[istate,k])**2     
                if bonding==0:
                    wgt_d8L2d10[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8L2d10[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8L2d10[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                    
               
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==0 and z3==z4==2 :
                if orb1=='d3z2r2' and orb2=='dx2y2' and S_Cu_12==0:
                    wgt_d10L2d8[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='dx2y2' and S_Cu_12==1:
                    wgt_d10L2d8[1]+=abs(vecs[istate,k])**2    
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and S_Cu_12==0:
                    wgt_d10L2d8[2]+=abs(vecs[istate,k])**2  
                elif orb1=='dx2y2' and orb2=='dx2y2' and S_Cu_12==0:
                    wgt_d10L2d8[3]+=abs(vecs[istate,k])**2                  
                if bonding==0:
                    wgt_d10L2d8[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d10L2d8[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d10L2d8[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==2 and z2==0 and z3==z4==2 :                               
                if orb1=='d3z2r2' and orb2=='dx2y2':
                    wgt_d9L2d9[0]+=abs(vecs[istate,k])**2 
                    if s3==s4 and s1=='up':
                        wgt_d9L2d9[4]+=abs(vecs[istate,k])**2
                    elif s3==s4 and s1=='dn':
                        wgt_d9L2d9[5]+=abs(vecs[istate,k])**2                        
                    elif s3!=s4 and s1=='up':
                        wgt_d9L2d9[6]+=abs(vecs[istate,k])**2
                    elif s3!=s4 and s1=='dn':
                        wgt_d9L2d9[7]+=abs(vecs[istate,k])**2                       
                       
                elif orb1=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9L2d9[1]+=abs(vecs[istate,k])**2            
                elif orb1=='d3z2r2' and orb2=='d3z2r2':
                    wgt_d9L2d9[2]+=abs(vecs[istate,k])**2    
                elif orb1=='dx2y2' and orb2=='dx2y2':
                    wgt_d9L2d9[3]+=abs(vecs[istate,k])**2  
                    if s3==s4 and s1=='up':
                        wgt_d9L2d9[8]+=abs(vecs[istate,k])**2
                    elif s3==s4 and s1=='dn':
                        wgt_d9L2d9[9]+=abs(vecs[istate,k])**2                        
                    elif s3!=s4 and s1=='up':
                        wgt_d9L2d9[10]+=abs(vecs[istate,k])**2
                    elif s3!=s4 and s1=='dn':
                        wgt_d9L2d9[11]+=abs(vecs[istate,k])**2    
                if bonding==0:
                    wgt_d9L2d9[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9L2d9[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9L2d9[48]+=abs(vecs[istate,k])**2                         
                        
                        
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==2 and z2==0 and z3==z4==0 :                               
                if orb1=='d3z2r2' and orb2=='dx2y2':
                    wgt_d9d9L2[0]+=abs(vecs[istate,k])**2    
                elif orb1=='dx2y2' and orb2=='d3z2r2':
                    wgt_d9d9L2[1]+=abs(vecs[istate,k])**2            
                elif orb1=='d3z2r2' and orb2=='d3z2r2':
                    wgt_d9d9L2[2]+=abs(vecs[istate,k])**2    
                elif orb1=='dx2y2' and orb2=='dx2y2':
                    wgt_d9d9L2[3]+=abs(vecs[istate,k])**2    
                if bonding==0:
                    wgt_d9d9L2[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9d9L2[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9d9L2[48]+=abs(vecs[istate,k])**2                     
                    
                    
                    
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.O_orbs) \
                      and (orb4 in pam.O_orbs) and z1==2 and z2==0 and z3==2 and z4==0 :     
             
                if orb1=='dx2y2' and orb2=='dx2y2' :
                    wgt_d9Ld9L[0]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2' and orb2=='dx2y2' :
                    wgt_d9Ld9L[1]+=abs(vecs[istate,k])**2   
                elif orb1=='dx2y2' and orb2=='d3z2r2' :
                    wgt_d9Ld9L[2]+=abs(vecs[istate,k])**2   
                elif orb1=='d3z2r2' and orb2=='d3z2r2' :
                    wgt_d9Ld9L[3]+=abs(vecs[istate,k])**2       
                    
                    
                if  orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==0 and S_Cu_12==0: 
                    wgt_d9Ld9L[4]+=abs(vecs[istate,k])**2
                    if i in indices[0]: 
                        print (' state ', i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,\
                           '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
                           ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
                           ", weight = ", weight,'\n')   
        
                    if s1==s2:
                        wgt_d9Ld9L[20]+=abs(vecs[istate,k])**2 
                    elif s1!=s2:
                        wgt_d9Ld9L[21]+=abs(vecs[istate,k])**2                          
                    
                elif  orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==1 and S_Cu_12==0: 
                    wgt_d9Ld9L[5]+=abs(vecs[istate,k])**2                    
                elif  orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==0 and S_Cu_12==1: 
                    wgt_d9Ld9L[6]+=abs(vecs[istate,k])**2                          
                elif  orb1=='dx2y2' and orb2=='dx2y2' and S_Ni_12==1 and S_Cu_12==1: 
                    wgt_d9Ld9L[7]+=abs(vecs[istate,k])**2                          
                elif  orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==0 and S_Cu_12==0: 
                    wgt_d9Ld9L[8]+=abs(vecs[istate,k])**2
                elif  orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==1 and S_Cu_12==0: 
                    wgt_d9Ld9L[9]+=abs(vecs[istate,k])**2   
                    if s1==s2:
                        wgt_d9Ld9L[22]+=abs(vecs[istate,k])**2 
                    elif s1!=s2:
                        wgt_d9Ld9L[23]+=abs(vecs[istate,k])**2                         
           
                elif  orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==0 and S_Cu_12==1: 
                    wgt_d9Ld9L[10]+=abs(vecs[istate,k])**2                          
                elif  orb1=='d3z2r2' and orb2=='dx2y2' and S_Ni_12==1 and S_Cu_12==1: 
                    wgt_d9Ld9L[11]+=abs(vecs[istate,k])**2                              
                elif  orb1=='dx2y2' and orb2=='d3z2r2' and S_Ni_12==0 and S_Cu_12==0: 
                    wgt_d9Ld9L[12]+=abs(vecs[istate,k])**2
                elif  orb1=='dx2y2' and orb2=='d3z2r2' and S_Ni_12==1 and S_Cu_12==0: 
                    wgt_d9Ld9L[13]+=abs(vecs[istate,k])**2                    
                elif  orb1=='dx2y2' and orb2=='d3z2r2' and S_Ni_12==0 and S_Cu_12==1: 
                    wgt_d9Ld9L[14]+=abs(vecs[istate,k])**2                          
                elif  orb1=='dx2y2' and orb2=='d3z2r2' and S_Ni_12==1 and S_Cu_12==1: 
                    wgt_d9Ld9L[15]+=abs(vecs[istate,k])**2                          
                elif  orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==0 and S_Cu_12==0: 
                    wgt_d9Ld9L[16]+=abs(vecs[istate,k])**2
                elif  orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==1 and S_Cu_12==0: 
                    wgt_d9Ld9L[17]+=abs(vecs[istate,k])**2                    
                elif  orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==0 and S_Cu_12==1: 
                    wgt_d9Ld9L[18]+=abs(vecs[istate,k])**2                          
                elif  orb1=='d3z2r2' and orb2=='d3z2r2' and S_Ni_12==1 and S_Cu_12==1: 
                    wgt_d9Ld9L[19]+=abs(vecs[istate,k])**2     
                if bonding==0:
                    wgt_d9Ld9L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9Ld9L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9Ld9L[48]+=abs(vecs[istate,k])**2                     

                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==2 and z2==z3==z4==0:                     
                if orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='d3z2r2' and S_Cu_12==0:
                    wgt_d9d8L[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==0:
                    wgt_d9d8L[1]+=abs(vecs[istate,k])**2                       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==0:
                    wgt_d9d8L[2]+=abs(vecs[istate,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==1:
                    wgt_d9d8L[3]+=abs(vecs[istate,k])**2       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==1:
                    wgt_d9d8L[4]+=abs(vecs[istate,k])**2                     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cu_12==0:
                    wgt_d9d8L[5]+=abs(vecs[istate,k])**2 
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cu_12==0:
                    wgt_d9d8L[6]+=abs(vecs[istate,k])**2                 
                if bonding==0:
                    wgt_d9d8L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9d8L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9d8L[48]+=abs(vecs[istate,k])**2              
            
            
            
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==2 and z3==z4==0:                 
                if orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2':
                    wgt_d8d9L[0]+=abs(vecs[istate,k])**2                                  
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='d3z2r2':
                    wgt_d8d9L[1]+=abs(vecs[istate,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2':
                    wgt_d8d9L[2]+=abs(vecs[istate,k])**2          
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2':
                    wgt_d8d9L[3]+=abs(vecs[istate,k])**2       
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2':
                    wgt_d8d9L[4]+=abs(vecs[istate,k])**2  
                if orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2':
                    if S_Ni_12==0 and S_Cu_12==0:
                        wgt_d8d9L[5]+=abs(vecs[istate,k])**2                               
                    elif S_Ni_12==1 and S_Cu_12==0:
                        wgt_d8d9L[6]+=abs(vecs[istate,k])**2 
                    elif S_Ni_12==0 and S_Cu_12==1:
                        wgt_d8d9L[7]+=abs(vecs[istate,k])**2                                
                    elif S_Ni_12==1 and S_Cu_12==1:
                        wgt_d8d9L[8]+=abs(vecs[istate,k])**2
                        if s1==s2 and s1==s3:
                            wgt_d8d9L[25]+=abs(vecs[istate,k])**2 
                        elif s1==s2 and s1!=s3:
                            wgt_d8d9L[26]+=abs(vecs[istate,k])**2                             
                        elif s1!=s2 and s1==s3:
                            wgt_d8d9L[27]+=abs(vecs[istate,k])**2     
                        elif s1!=s2 and s1!=s3:
                            wgt_d8d9L[28]+=abs(vecs[istate,k])**2         
                                
                if orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='d3z2r2': 
                    if S_Ni_12==0 and S_Cu_12==0:
                        wgt_d8d9L[9]+=abs(vecs[istate,k])**2                               
                    elif S_Ni_12==1 and S_Cu_12==0:
                        wgt_d8d9L[10]+=abs(vecs[istate,k])**2        
                    elif S_Ni_12==0 and S_Cu_12==1:
                        wgt_d8d9L[11]+=abs(vecs[istate,k])**2                                
                    elif S_Ni_12==1 and S_Cu_12==1:
                        wgt_d8d9L[12]+=abs(vecs[istate,k])**2                        
                if orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2':
                    if S_Ni_12==0 and S_Cu_12==0:
                        wgt_d8d9L[13]+=abs(vecs[istate,k])**2                               
                    elif S_Ni_12==1 and S_Cu_12==0:
                        wgt_d8d9L[14]+=abs(vecs[istate,k])**2 
                    elif S_Ni_12==0 and S_Cu_12==1:
                        wgt_d8d9L[15]+=abs(vecs[istate,k])**2                                
                    elif S_Ni_12==1 and S_Cu_12==1:
                        wgt_d8d9L[16]+=abs(vecs[istate,k])**2             
                if orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2':
                    if S_Ni_12==0 and S_Cu_12==0:
                        wgt_d8d9L[17]+=abs(vecs[istate,k])**2                               
                    elif S_Ni_12==1 and S_Cu_12==0:
                        wgt_d8d9L[18]+=abs(vecs[istate,k])**2 
                        if s1==s3:
                            wgt_d8d9L[29]+=abs(vecs[istate,k])**2 
                        elif s1!=s3:
                            wgt_d8d9L[30]+=abs(vecs[istate,k])**2                                  
                        
                    elif S_Ni_12==0 and S_Cu_12==1:
                        wgt_d8d9L[19]+=abs(vecs[istate,k])**2                                
                    elif S_Ni_12==1 and S_Cu_12==1:
                        wgt_d8d9L[20]+=abs(vecs[istate,k])**2         
                if orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2':
                    if S_Ni_12==0 and S_Cu_12==0:
                        wgt_d8d9L[21]+=abs(vecs[istate,k])**2                               
                    elif S_Ni_12==1 and S_Cu_12==0:
                        wgt_d8d9L[22]+=abs(vecs[istate,k])**2        
                    elif S_Ni_12==0 and S_Cu_12==1:
                        wgt_d8d9L[23]+=abs(vecs[istate,k])**2                                
                    elif S_Ni_12==1 and S_Cu_12==1:
                        wgt_d8d9L[24]+=abs(vecs[istate,k])**2                            
                if bonding==0:
                    wgt_d8d9L[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8d9L[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8d9L[48]+=abs(vecs[istate,k])**2 
                    
                    
                    

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z4==2 and z2==z3==0:                 
                if orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='d3z2r2' and S_Cu_12==0:
                    wgt_d9Ld8[0]+=abs(vecs[istate,k])**2                      
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' :
                    wgt_d9Ld8[1]+=abs(vecs[istate,k])**2                       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==0:
                    wgt_d9Ld8[2]+=abs(vecs[istate,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==1:
                    wgt_d9Ld8[3]+=abs(vecs[istate,k])**2       
                elif orb1=='dx2y2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Cu_12==1:
                    wgt_d9Ld8[4]+=abs(vecs[istate,k])**2                     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cu_12==0:
                    wgt_d9Ld8[5]+=abs(vecs[istate,k])**2 
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2':
                    wgt_d9Ld8[6]+=abs(vecs[istate,k])**2                 
                if bonding==0:
                    wgt_d9Ld8[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d9Ld8[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d9Ld8[48]+=abs(vecs[istate,k])**2 
                    
                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.O_orbs) and z1==z2==z4==2 and z3==0:                      
                if orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Ni_12==0:
                    wgt_d8Ld9[0]+=abs(vecs[istate,k])**2                       
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Ni_12==1:
                    wgt_d8Ld9[1]+=abs(vecs[istate,k])**2                       
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Ni_12==0:
                    wgt_d8Ld9[2]+=abs(vecs[istate,k])**2                       
                elif orb1=='d3z2r2' and orb2=='d3z2r2'  and  orb3=='dx2y2' and S_Ni_12==0:
                    wgt_d8Ld9[3]+=abs(vecs[istate,k])**2     
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Ni_12==1:
                    wgt_d8Ld9[4]+=abs(vecs[istate,k])**2       
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Ni_12==0:
                    wgt_d8Ld9[5]+=abs(vecs[istate,k])**2   
                elif orb1=='dx2y2' and orb2=='dx2y2'  and  orb3=='dx2y2' and S_Ni_12==0:
                    wgt_d8Ld9[6]+=abs(vecs[istate,k])**2                   
                if bonding==0:
                    wgt_d8Ld9[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8Ld9[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8Ld9[48]+=abs(vecs[istate,k])**2 
                    
                    
                    

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs)  and  (orb3 in pam.Ni_Cu_orbs) \
                      and (orb4 in pam.Ni_Cu_orbs) and z1==z2==2 and z3==z4==0:                      
                if orb1=='dx2y2' and orb2=='dx2y2'  and orb3=='d3z2r2' and orb4=='d3z2r2':
                    wgt_d8d8[0]+=abs(vecs[istate,k])**2                       
                elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='d3z2r2' and orb4=='dx2y2':
                    wgt_d8d8[1]+=abs(vecs[istate,k])**2 
                    if S_Ni_12==0 and S_Cu_12==0:
                        wgt_d8d8[6]+=abs(vecs[istate,k])**2                               
                    elif S_Ni_12==1 and S_Cu_12==0:
                        wgt_d8d8[7]+=abs(vecs[istate,k])**2        
                    elif S_Ni_12==0 and S_Cu_12==1:
                        wgt_d8d8[8]+=abs(vecs[istate,k])**2                                
                    elif S_Ni_12==1 and S_Cu_12==1:
                        wgt_d8d8[9]+=abs(vecs[istate,k])**2 
                        if s1==s2 and s1==s3:
                            wgt_d8d8[10]+=abs(vecs[istate,k])**2 
                        elif s1==s2 and s1!=s3:
                            wgt_d8d8[11]+=abs(vecs[istate,k])**2                             
                        elif s1!=s2 and s1==s3:
                            wgt_d8d8[12]+=abs(vecs[istate,k])**2     
                        elif s1!=s2 and s1!=s3:
                            wgt_d8d8[13]+=abs(vecs[istate,k])**2                                 
                            
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='dx2y2' and orb4=='dx2y2':
                    wgt_d8d8[2]+=abs(vecs[istate,k])**2                          
                elif orb1=='dx2y2' and orb2=='dx2y2'  and orb3=='dx2y2' and orb4=='dx2y2':
                    wgt_d8d8[3]+=abs(vecs[istate,k])**2  
                elif orb1=='d3z2r2' and orb2=='dx2y2'  and orb3=='dx2y2' and orb4=='dx2y2':
                    wgt_d8d8[4]+=abs(vecs[istate,k])**2                  
                elif orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='d3z2r2' and orb4=='d3z2r2':
                    wgt_d8d8[5]+=abs(vecs[istate,k])**2  
                if bonding==0:
                    wgt_d8d8[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_d8d8[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_d8d8[48]+=abs(vecs[istate,k])**2 
                    
                    
                    
                    
                    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs):
                wgt_d8d8[14]+=abs(vecs[istate,k])**2 
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==2 and z3==z4==0:
                wgt_d9Ld10L2[12]+=abs(vecs[istate,k])**2
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==2 and z2==z3==z4==0:
                wgt_d9d10L3[8]+=abs(vecs[istate,k])**2    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==0 and z2==z3==z4==2:
                wgt_d10L3d9[10]+=abs(vecs[istate,k])**2                    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==2 and z4==0:
                wgt_d9L2d10L[8]+=abs(vecs[istate,k])**2
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z2==2 and z1==z3==z4==0:
                wgt_d10Ld9L2[8]+=abs(vecs[istate,k])**2      
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==z4==0:
                wgt_d10d9L3[8]+=abs(vecs[istate,k])**2                     
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z2==z3==2 and z1==z4==0:
                wgt_d10L2d9L[8]+=abs(vecs[istate,k])**2  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==2 and z4==0:
                wgt_d8Ld10L[8]+=abs(vecs[istate,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==2 and z3==z4==0:
                wgt_d8d10L2[8]+=abs(vecs[istate,k])**2  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z3==2 and z1==z2==z4==0:
                wgt_d10Ld8L[8]+=abs(vecs[istate,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==z4==0:
                wgt_d10d8L2[8]+=abs(vecs[istate,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z3==z4==2:
                wgt_d8L2d10[8]+=abs(vecs[istate,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z3==z4==2 and z1==z2==0:
                wgt_d10L2d8[8]+=abs(vecs[istate,k])**2                    
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z3==z4==2 and z2==0:
                wgt_d9L2d9[12]+=abs(vecs[istate,k])**2                  
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==2 and z2==z3==z4==0:
                wgt_d9d9L2[8]+=abs(vecs[istate,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
                and z1==z3==2 and z2==z4==0:
                wgt_d9Ld9L[38]+=abs(vecs[istate,k])**2                     
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==2 and z2==z3==z4==0:
                wgt_d9d8L[8]+=abs(vecs[istate,k])**2   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==2 and z3==z4==0:
                wgt_d8d9L[31]+=abs(vecs[istate,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==z4==2 and z2==z3==0:
                wgt_d9Ld8[8]+=abs(vecs[istate,k])**2                   
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) \
                and z1==z2==z4==2 and z3==0:
                wgt_d8Ld9[8]+=abs(vecs[istate,k])**2                   

#             if (orb1 =='dx2y2') and (orb2 =='dx2y2') and (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) \
#                 and z1==z3==1 and z2==z4==0:
#                 print (' state ', i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,\
#                        '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
#                        ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
#                        ", weight = ", weight,'\n')  
#                 if S_Ni_12==1 and S_Cu_12==1:
#                     s11+=abs(vecs[istate,k])**2 
#                 if S_Ni_12==1 and S_Cu_12==0:
#                     s10+=abs(vecs[istate,k])**2                     
#                 if S_Ni_12==0 and S_Cu_12==1:
#                     s01+=abs(vecs[istate,k])**2                     
#                 if S_Ni_12==0 and S_Cu_12==0:
#                     s00+=abs(vecs[istate,k])**2 
                    
            if (orb1 in pam.Obilayer_orbs) or (orb2 in pam.Obilayer_orbs) or (orb3 in pam.Obilayer_orbs) or (orb4 in pam.Obilayer_orbs):
                wgt_H[0]+=abs(vecs[istate,k])**2    
                
                
            if (orb1 in pam.Obilayer_orbs):    
                wgt_s[0]+=abs(vecs[istate,k])**2      
                if bonding==0:
                    wgt_s[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_s[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_s[48]+=abs(vecs[istate,k])**2                 
                
                
                
            elif (orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs and orb3 in pam.Obilayer_orbs and orb4 in pam.Obilayer_orbs): 
                wgt_ddss[0]+=abs(vecs[istate,k])**2  
                if z1==z2==2:
                    wgt_ddss[1]+=abs(vecs[istate,k])**2 
                    if orb1=='d3z2r2' and orb2=='d3z2r2':
                        wgt_ddss[4]+=abs(vecs[istate,k])**2  
                    elif orb1=='d3z2r2' and orb2=='dx2y2':
                        wgt_ddss[5]+=abs(vecs[istate,k])**2                          
                    elif orb1=='dx2y2' and orb2=='dx2y2':
                        wgt_ddss[6]+=abs(vecs[istate,k])**2                                                   
                elif z1==z2==0:
                    wgt_ddss[2]+=abs(vecs[istate,k])**2  
                    if orb1=='d3z2r2' and orb2=='d3z2r2':
                        wgt_ddss[7]+=abs(vecs[istate,k])**2  
                    elif orb1=='d3z2r2' and orb2=='dx2y2':
                        wgt_ddss[8]+=abs(vecs[istate,k])**2                          
                    elif orb1=='dx2y2' and orb2=='dx2y2':
                        wgt_ddss[9]+=abs(vecs[istate,k])**2                        
                elif z1==2 and z2==0:
                    wgt_ddss[3]+=abs(vecs[istate,k])**2  
                    if orb1=='d3z2r2' and orb2=='d3z2r2':
                        wgt_ddss[10]+=abs(vecs[istate,k])**2  
                    elif orb1=='d3z2r2' and orb2=='dx2y2':
                        wgt_ddss[11]+=abs(vecs[istate,k])**2                          
                    elif orb1=='dx2y2' and orb2=='dx2y2':
                        wgt_ddss[12]+=abs(vecs[istate,k])**2  
                if bonding==0:
                    wgt_ddss[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_ddss[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_ddss[48]+=abs(vecs[istate,k])**2                         
                        
                        
                        
                    
            elif (orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs and orb3 in pam.Obilayer_orbs and orb4 in pam.O_orbs): 
                wgt_ddsp[0]+=abs(vecs[istate,k])**2   
                if z1==z2==2:
                    wgt_ddsp[1]+=abs(vecs[istate,k])**2 
                    if orb1=='d3z2r2' and orb2=='d3z2r2':
                        wgt_ddsp[4]+=abs(vecs[istate,k])**2  
                    elif orb1=='d3z2r2' and orb2=='dx2y2':
                        wgt_ddsp[5]+=abs(vecs[istate,k])**2                          
                    elif orb1=='dx2y2' and orb2=='dx2y2':
                        wgt_ddsp[6]+=abs(vecs[istate,k])**2                                                   
                elif z1==z2==0:
                    wgt_ddsp[2]+=abs(vecs[istate,k])**2  
                    if orb1=='d3z2r2' and orb2=='d3z2r2':
                        wgt_ddsp[7]+=abs(vecs[istate,k])**2  
                    elif orb1=='d3z2r2' and orb2=='dx2y2':
                        wgt_ddsp[8]+=abs(vecs[istate,k])**2                          
                    elif orb1=='dx2y2' and orb2=='dx2y2':
                        wgt_ddsp[9]+=abs(vecs[istate,k])**2                        
                elif z1==2 and z2==0:
                    wgt_ddsp[3]+=abs(vecs[istate,k])**2 
                    if z4==2:
                        wgt_ddsp[17]+=abs(vecs[istate,k])**2
                    if z4==0:
                        wgt_ddsp[18]+=abs(vecs[istate,k])**2                        
                    if orb1=='d3z2r2' and orb2=='d3z2r2':
                        wgt_ddsp[10]+=abs(vecs[istate,k])**2  
                    elif orb1=='d3z2r2' and orb2=='dx2y2':
                        wgt_ddsp[11]+=abs(vecs[istate,k])**2 
                        if z4==2:
                            wgt_ddsp[13]+=abs(vecs[istate,k])**2  
                            if S_Ni_12==1:
                                 wgt_ddsp[15]+=abs(vecs[istate,k])**2
                            if S_Ni_12==0:
                                 wgt_ddsp[16]+=abs(vecs[istate,k])**2                                  
                        if z4==0:
                            wgt_ddsp[14]+=abs(vecs[istate,k])**2                              
                    elif orb1=='dx2y2' and orb2=='dx2y2':
                        wgt_ddsp[12]+=abs(vecs[istate,k])**2     
                if bonding==0:
                    wgt_ddsp[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_ddsp[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_ddsp[48]+=abs(vecs[istate,k])**2                         
                        
                        
                        
                        
            elif (orb1 in pam.Ni_Cu_orbs and orb2 in pam.Obilayer_orbs and orb3 in pam.Obilayer_orbs): 
                wgt_dssp[0]+=abs(vecs[istate,k])**2   
                if z1==2 and z4==2:
                    wgt_dssp[1]+=abs(vecs[istate,k])**2
                    if orb1=='d3z2r2':
                        wgt_dssp[3]+=abs(vecs[istate,k])**2                        
                    elif orb1=='dx2y2':
                        wgt_dssp[4]+=abs(vecs[istate,k])**2                        
                elif z1==2 and z4==0:
                    wgt_dssp[2]+=abs(vecs[istate,k])**2
                    if orb1=='d3z2r2':
                        wgt_dssp[5]+=abs(vecs[istate,k])**2                        
                    elif orb1=='dx2y2':
                        wgt_dssp[6]+=abs(vecs[istate,k])**2      
                if bonding==0:
                    wgt_dssp[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_dssp[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_dssp[48]+=abs(vecs[istate,k])**2                         
                        
                        
                    
                
            elif (orb1 in pam.Ni_Cu_orbs and orb2 in pam.Obilayer_orbs and orb3 in pam.O_orbs): 
                wgt_dspp[0]+=abs(vecs[istate,k])**2                
                if z1==2 and z3==2 and z4==0:
                    wgt_dspp[1]+=abs(vecs[istate,k])**2
                    if orb1=='d3z2r2':
                        wgt_dspp[3]+=abs(vecs[istate,k])**2                        
                    elif orb1=='dx2y2':
                        wgt_dspp[4]+=abs(vecs[istate,k])**2                        
                elif z1==2 and z3==0 and z4==0:
                    wgt_dspp[2]+=abs(vecs[istate,k])**2
                    if orb1=='d3z2r2':
                        wgt_dspp[5]+=abs(vecs[istate,k])**2                        
                    elif orb1=='dx2y2':
                        wgt_dspp[6]+=abs(vecs[istate,k])**2   
                if bonding==0:
                    wgt_dspp[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_dspp[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_dspp[48]+=abs(vecs[istate,k])**2                         
                        
                        
                
            elif (orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs and orb3 in pam.Ni_Cu_orbs and orb4 in pam.Obilayer_orbs): 
                wgt_ddds[0]+=abs(vecs[istate,k])**2 
                if z1==z2==2 and z3==0:
                    wgt_ddds[1]+=abs(vecs[istate,k])**2        
                    if orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='d3z2r2':
                        wgt_ddds[3]+=abs(vecs[istate,k])**2       
                    elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='d3z2r2':
                        wgt_ddds[4]+=abs(vecs[istate,k])**2                               
                    elif orb1=='dx2y2' and orb2=='dx2y2' and orb3=='d3z2r2':
                        wgt_ddds[5]+=abs(vecs[istate,k])**2                             
                    elif orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='dx2y2':
                        wgt_ddds[6]+=abs(vecs[istate,k])**2       
                    elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='dx2y2':
                        wgt_ddds[7]+=abs(vecs[istate,k])**2
                        if S_Ni_12==1:
                             wgt_ddds[15]+=abs(vecs[istate,k])**2
                        if S_Ni_12==0:
                             wgt_ddds[16]+=abs(vecs[istate,k])**2                                
                    elif orb1=='dx2y2' and orb2=='dx2y2' and orb3=='dx2y2':
                        wgt_ddds[8]+=abs(vecs[istate,k])**2                          
                elif z1==2 and z3==z2==0:
                    wgt_ddds[2]+=abs(vecs[istate,k])**2          
                    if orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='d3z2r2':
                        wgt_ddds[9]+=abs(vecs[istate,k])**2                         
                    elif orb1=='d3z2r2' and orb2=='d3z2r2' and orb3=='dx2y2':
                        wgt_ddds[10]+=abs(vecs[istate,k])**2                        
                    elif orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='dx2y2':
                        wgt_ddds[11]+=abs(vecs[istate,k])**2  
                    elif orb1=='dx2y2' and orb2=='d3z2r2' and orb3=='d3z2r2':
                        wgt_ddds[12]+=abs(vecs[istate,k])**2                         
                    elif orb1=='dx2y2' and orb2=='d3z2r2' and orb3=='dx2y2':
                        wgt_ddds[13]+=abs(vecs[istate,k])**2                        
                    elif orb1=='dx2y2' and orb2=='dx2y2' and orb3=='dx2y2':
                        wgt_ddds[14]+=abs(vecs[istate,k])**2                          
                if bonding==0:
                    wgt_ddds[46]+=abs(vecs[istate,k])**2 
                if bonding==1:
                    wgt_ddds[47]+=abs(vecs[istate,k])**2 
                if bonding==-1:
                    wgt_ddds[48]+=abs(vecs[istate,k])**2                         
                        
                
            sumweight=sumweight+abs(vecs[istate,k])**2

      
#         print ('s11=',s11)        
#         print ('s10=',s10)       
#         print ('s01=',s01)  
#         print ('s00=',s00)          


        wgt_d9Ld10L2[13]=wgt_d9Ld10L2[0]+wgt_d9Ld10L2[1]        
        wgt_d9d10L3[2]=wgt_d9d10L3[0]+wgt_d9d10L3[1]          
        wgt_d9L2d10L[2]=wgt_d9L2d10L[0]+wgt_d9L2d10L[1]        
        wgt_d10Ld9L2[2]=wgt_d10Ld9L2[0]+wgt_d10Ld9L2[1]  
        wgt_d10d9L3[2]=wgt_d10d9L3[0]+wgt_d10d9L3[1]        
        wgt_d10L2d9L[2]=wgt_d10L2d9L[0]+wgt_d10L2d9L[1]          
        wgt_d8Ld10L[4]=wgt_d8Ld10L[0]+wgt_d8Ld10L[1]+wgt_d8Ld10L[2]+wgt_d8Ld10L[3]        
        wgt_d10Ld8L[4]=wgt_d10Ld8L[0]+wgt_d10Ld8L[1]+wgt_d10Ld8L[2]+wgt_d10Ld8L[3]    
        wgt_d8d10L2[4]=wgt_d8d10L2[0]+wgt_d8d10L2[1]+wgt_d8d10L2[2]+wgt_d8d10L2[3]          
        wgt_d10d8L2[4]=wgt_d10d8L2[0]+wgt_d10d8L2[1]+wgt_d10d8L2[2]+wgt_d10d8L2[3]            
        wgt_d8L2d10[4]=wgt_d8L2d10[0]+wgt_d8L2d10[1]+wgt_d8L2d10[2]+wgt_d8L2d10[3]          
        wgt_d10L2d8[4]=wgt_d10L2d8[0]+wgt_d10L2d8[1]+wgt_d10L2d8[2]+wgt_d10L2d8[3]            
        wgt_d9L2d9[13]=wgt_d9L2d9[0]+wgt_d9L2d9[1]+wgt_d9L2d9[2]+wgt_d9L2d9[3]
        wgt_d9d9L2[4]=wgt_d9d9L2[0]+wgt_d9d9L2[1]+wgt_d9d9L2[2]+wgt_d9d9L2[3]
        wgt_d9Ld9L[39]=wgt_d9Ld9L[0]+wgt_d9Ld9L[1]+wgt_d9Ld9L[2]+wgt_d9Ld9L[3]

        wgt_d9d8L[7]=wgt_d9d8L[0]+wgt_d9d8L[1]+wgt_d9d8L[2]+wgt_d9d8L[3]+wgt_d9d8L[4]+wgt_d9d8L[5] +wgt_d9d8L[6]        
        wgt_d8d9L[32]=wgt_d8d9L[0]+wgt_d8d9L[1]+wgt_d8d9L[2]+wgt_d8d9L[3]+wgt_d8d9L[4] 
        wgt_d9Ld8[7]=wgt_d9Ld8[0]+wgt_d9Ld8[1]+wgt_d9Ld8[2]+wgt_d9Ld8[3]+wgt_d9Ld8[4]+wgt_d9Ld8[5] +wgt_d9Ld8[6]     
        wgt_d8Ld9[7]=wgt_d8Ld9[0]+wgt_d8Ld9[1]+wgt_d8Ld9[2]+wgt_d8Ld9[3]+wgt_d8Ld9[4]+wgt_d8Ld9[5] +wgt_d8Ld9[6] 


        sumweight_picture=wgt_LmLn[0]+wgt_d9Ld10L2[12]+wgt_d9d10L3[2]+wgt_d10L3d9[10]+wgt_d9L2d10L[2]+wgt_d10Ld9L2[2]+wgt_d10d9L3[2]\
                    +wgt_d10L2d9L[2]+wgt_d8Ld10L[4]+wgt_d10Ld8L[4]+wgt_d8d10L2[4]+wgt_d10d8L2[4]+wgt_d8L2d10[4]\
                    +wgt_d10L2d8[4]+wgt_d9L2d9[12]+wgt_d9d9L2[4]+wgt_d9Ld9L[39]+wgt_d9d8L[7]+wgt_d8d9L[32]\
                    +wgt_d9Ld8[7]+wgt_d8Ld9[7]+ wgt_d8d8[14] 
        sumweight2=wgt_LmLn[0]+wgt_d9Ld10L2[13]+wgt_d9d10L3[8]+wgt_d10L3d9[10]+wgt_d9L2d10L[8]+wgt_d10Ld9L2[8]+wgt_d10d9L3[8]\
                          +wgt_d10L2d9L[8]+wgt_d8Ld10L[8]+wgt_d10Ld8L[8]+wgt_d8d10L2[8]+wgt_d10d8L2[8]\
                          +wgt_d8L2d10[8]+ wgt_d10L2d8[8]+wgt_d9L2d9[13]+wgt_d9d9L2[8]+wgt_d9Ld9L[38]\
                          +wgt_d9d8L[8]+wgt_d8d9L[32]+wgt_d9Ld8[8]+wgt_d8Ld9[8]+ wgt_d8d8[14]

    print ('sumweight=',sumweight/number)
    #         print ('sumweigh1=',sumweigh1/number)
    #         print ('OBILAYER=',wgt_OBILAYER[0]) 

    print ('s=',wgt_s[0]/number,'  bonding0=',wgt_s[46]/number,'  bonding1=',wgt_s[47]/number,'  bonding-1=',wgt_s[48]/number) 
    print ('ddss=',wgt_ddss[0]/number,'  bonding0=',wgt_ddss[46]/number,'  bonding1=',wgt_ddss[47]/number,'  bonding-1=',wgt_ddss[48]/number)  
    print ('ddsp=',wgt_ddsp[0]/number,'  bonding0=',wgt_ddsp[46]/number,'  bonding1=',wgt_ddsp[47]/number,'  bonding-1=',wgt_ddsp[48]/number) 
    print ('dssp=',wgt_dssp[0]/number,'  bonding0=',wgt_dssp[46]/number,'  bonding1=',wgt_dssp[47]/number,'  bonding-1=',wgt_dssp[48]/number) 
    print ('dspp=',wgt_dspp[0]/number,'  bonding0=',wgt_dspp[46]/number,'  bonding1=',wgt_dspp[47]/number,'  bonding-1=',wgt_dspp[48]/number)
    print ('ddds=',wgt_ddds[0]/number,'  bonding0=',wgt_ddds[46]/number,'  bonding1=',wgt_ddds[47]/number,'  bonding-1=',wgt_ddds[48]/number) 


    path = './data'		# create file

    if os.path.isdir(path) == False:
        os.mkdir(path) 

        
    txt=open('./data/a1','a')                                  
    txt.write(str(wgt_a1[0]/number)+'\n')
    txt.close()           
    txt=open('./data/b1','a')                                  
    txt.write(str(wgt_b1[0]/number)+'\n')
    txt.close()     
    txt=open('./data/L','a')                                  
    txt.write(str(wgt_L[0]/number)+'\n')
    txt.close()         
    txt=open('./data/O','a')                                  
    txt.write(str(wgt_O[0]/number)+'\n')
    txt.close()            
        
        
    txt=open('./data/LmLn','a')                                  
    txt.write(str(wgt_LmLn[0]/number)+'\n')
    txt.close()     
    txt=open('./data/LmLn_00','a')                                  
    txt.write(str(wgt_LmLn[1]/number)+'\n')
    txt.close()   
    txt=open('./data/LmLn_10','a')                                  
    txt.write(str(wgt_LmLn[2]/number)+'\n')
    txt.close()           
    txt=open('./data/LmLn_01','a')                                  
    txt.write(str(wgt_LmLn[3]/number)+'\n')
    txt.close()           
    txt=open('./data/LmLn_11','a')                                  
    txt.write(str(wgt_LmLn[4]/number)+'\n')
    txt.close()           

    txt=open('./data/d9Ld10L2_dx2y2','a')                                  
    txt.write(str(wgt_d9Ld10L2[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9Ld10L2_d3z2r2','a')                                  
    txt.write(str(wgt_d9Ld10L2[1]/number)+'\n')
    txt.close() 
    txt=open('./data/d9Ld10L2','a')                                  
    txt.write(str(wgt_d9Ld10L2[12]/number)+'\n')
    txt.close() 
    txt=open('./data/d9Ld10L2_dx2y2_00','a')                                  
    txt.write(str(wgt_d9Ld10L2[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld10L2_dx2y2_10','a')                                  
    txt.write(str(wgt_d9Ld10L2[3]/number)+'\n')
    txt.close()                  
    txt=open('./data/d9Ld10L2_dx2y2_01','a')                                  
    txt.write(str(wgt_d9Ld10L2[4]/number)+'\n')
    txt.close()                  
    txt=open('./data/d9Ld10L2_dx2y2_11','a')                                  
    txt.write(str(wgt_d9Ld10L2[5]/number)+'\n')
    txt.close()                  
    txt=open('./data/d9Ld10L2_d3z2r2_00','a')                                  
    txt.write(str(wgt_d9Ld10L2[6]/number)+'\n')
    txt.close()         
    txt=open('./data/d9Ld10L2_d3z2r2_10','a')                                  
    txt.write(str(wgt_d9Ld10L2[7]/number)+'\n')
    txt.close()    
    txt=open('./data/d9Ld10L2_d3z2r2_01','a')                                  
    txt.write(str(wgt_d9Ld10L2[8]/number)+'\n')
    txt.close()            
    txt=open('./data/d9Ld10L2_d3z2r2_11','a')                                  
    txt.write(str(wgt_d9Ld10L2[9]/number)+'\n')
    txt.close()     
    txt=open('./data/d9Ld10L2_u','a')                                  
    txt.write(str(wgt_d9Ld10L2[10]/number)+'\n')
    txt.close()             
    txt=open('./data/d9Ld10L2_d','a')                                  
    txt.write(str(wgt_d9Ld10L2[11]/number)+'\n')
    txt.close()             
    txt=open('./data/d9Ld10L2+d10L2d9L','a')                                  
    txt.write(str(wgt_d9Ld10L2[12]/number+wgt_d10L2d9L[2]/number)+'\n')
    txt.close() 


    txt=open('./data/d9d10L3_dx2y2','a')                                  
    txt.write(str(wgt_d9d10L3[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9d10L3_d3z2r2','a')                                  
    txt.write(str(wgt_d9d10L3[1]/number)+'\n')
    txt.close()
    txt=open('./data/d9d10L3','a')                                  
    txt.write(str(wgt_d9d10L3[2]/number)+'\n')
    txt.close()   

    txt=open('./data/d10L3d9_dx2y2','a')                                  
    txt.write(str(wgt_d10L3d9[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10L3d9_d3z2r2','a')                                  
    txt.write(str(wgt_d10L3d9[1]/number)+'\n')
    txt.close() 
    txt=open('./data/d10L3d9','a')                                  
    txt.write(str(wgt_d10L3d9[10]/number)+'\n')
    txt.close() 
    txt=open('./data/d10L3d9_dx2y2_u','a')                                  
    txt.write(str(wgt_d10L3d9[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d10L3d9_dx2y2_d','a')                                  
    txt.write(str(wgt_d10L3d9[3]/number)+'\n')
    txt.close()                  
    txt=open('./data/d10L3d9_d3z2r2_u','a')                                  
    txt.write(str(wgt_d10L3d9[4]/number)+'\n')
    txt.close()         
    txt=open('./data/d10L3d9_d3z2r2_d','a')                                  
    txt.write(str(wgt_d10L3d9[5]/number)+'\n')
    txt.close()    





    txt=open('./data/d9L2d10L_dx2y2','a')                                  
    txt.write(str(wgt_d9L2d10L[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9L2d10L_d3z2r2','a')                                  
    txt.write(str(wgt_d9L2d10L[1]/number)+'\n')
    txt.close()
    txt=open('./data/d9L2d10L','a')                                  
    txt.write(str(wgt_d9L2d10L[2]/number)+'\n')
    txt.close()        

    txt=open('./data/d10Ld9L2_dx2y2','a')                                  
    txt.write(str(wgt_d10Ld9L2[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10Ld9L2_d3z2r2','a')                                  
    txt.write(str(wgt_d10Ld9L2[1]/number)+'\n')
    txt.close()
    txt=open('./data/d10Ld9L2','a')                                  
    txt.write(str(wgt_d10Ld9L2[2]/number)+'\n')
    txt.close()        

    txt=open('./data/d10d9L3_dx2y2','a')                                  
    txt.write(str(wgt_d10d9L3[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10d9L3_d3z2r2','a')                                  
    txt.write(str(wgt_d10d9L3[1]/number)+'\n')
    txt.close()    
    txt=open('./data/d10d9L3','a')                                  
    txt.write(str(wgt_d10d9L3[2]/number)+'\n')
    txt.close()         

    txt=open('./data/d10L2d9L_dx2y2','a')                                  
    txt.write(str(wgt_d10L2d9L[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10L2d9L_d3z2r2','a')                                  
    txt.write(str(wgt_d10L2d9L[1]/number)+'\n')
    txt.close()   
    txt=open('./data/d10L2d9L','a')                                  
    txt.write(str(wgt_d10L2d9L[2]/number)+'\n')
    txt.close()          

    txt=open('./data/d8Ld10L_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d8Ld10L[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d8Ld10L_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d8Ld10L[1]/number)+'\n')
    txt.close() 
    txt=open('./data/d8Ld10L_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d8Ld10L[2]/number)+'\n')
    txt.close()        
    txt=open('./data/d8Ld10L_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8Ld10L[3]/number)+'\n')
    txt.close()        
    txt=open('./data/d8Ld10L','a')                                  
    txt.write(str(wgt_d8Ld10L[4]/number)+'\n')
    txt.close()              

    txt=open('./data/d10Ld8L_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d10Ld8L[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10Ld8L_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d10Ld8L[1]/number)+'\n')
    txt.close() 
    txt=open('./data/d10Ld8L_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d10Ld8L[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d10Ld8L_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d10Ld8L[3]/number)+'\n')
    txt.close()          
    txt=open('./data/d10Ld8L','a')                                  
    txt.write(str(wgt_d10Ld8L[4]/number)+'\n')
    txt.close()               

    txt=open('./data/d8d10L2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d8d10L2[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d8d10L2_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d8d10L2[1]/number)+'\n')
    txt.close() 
    txt=open('./data/d8d10L2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d8d10L2[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d10L2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8d10L2[3]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d10L2','a')                                  
    txt.write(str(wgt_d8d10L2[4]/number)+'\n')
    txt.close()         

    txt=open('./data/d10d8L2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d10d8L2[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10d8L2_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d10d8L2[1]/number)+'\n')
    txt.close()  
    txt=open('./data/d10d8L2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d10d8L2[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d10d8L2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d10d8L2[3]/number)+'\n')
    txt.close()          
    txt=open('./data/d10d8L2','a')                                  
    txt.write(str(wgt_d10d8L2[4]/number)+'\n')
    txt.close()         

    txt=open('./data/d8L2d10_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d8L2d10[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d8L2d10_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d8L2d10[1]/number)+'\n')
    txt.close()
    txt=open('./data/d8L2d10_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d8L2d10[2]/number)+'\n')
    txt.close()         
    txt=open('./data/d8L2d10_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8L2d10[3]/number)+'\n')
    txt.close()         
    txt=open('./data/d8L2d10','a')                                  
    txt.write(str(wgt_d8L2d10[4]/number)+'\n')
    txt.close()         

    txt=open('./data/d10L2d8_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d10L2d8[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d10L2d8_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d10L2d8[1]/number)+'\n')
    txt.close()
    txt=open('./data/d10L2d8_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d10L2d8[2]/number)+'\n')
    txt.close()         
    txt=open('./data/d10L2d8_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d10L2d8[3]/number)+'\n')
    txt.close()         
    txt=open('./data/d10L2d8','a')                                  
    txt.write(str(wgt_d10L2d8[4]/number)+'\n')
    txt.close()         

    txt=open('./data/d9L2d9_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d9L2d9[0]/number)+'\n')
    txt.close()  
    txt=open('./data/d9L2d9_dx2y2_d3z2r2','a')                                  
    txt.write(str(wgt_d9L2d9[1]/number)+'\n')
    txt.close()   
    txt=open('./data/d9L2d9_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d9L2d9[2]/number)+'\n')
    txt.close()  
    txt=open('./data/d9L2d9_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d9L2d9[3]/number)+'\n')
    txt.close()           
    txt=open('./data/d9L2d9_d3z2r2_dx2y2_su','a')                                  
    txt.write(str(wgt_d9L2d9[4]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9_d3z2r2_dx2y2_sd','a')                                  
    txt.write(str(wgt_d9L2d9[5]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9_d3z2r2_dx2y2_du','a')                                  
    txt.write(str(wgt_d9L2d9[6]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9_d3z2r2_dx2y2_dd','a')                                  
    txt.write(str(wgt_d9L2d9[7]/number)+'\n')
    txt.close()                  
    txt=open('./data/d9L2d9_dx2y2_dx2y2_su','a')                                  
    txt.write(str(wgt_d9L2d9[8]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9_dx2y2_dx2y2_sd','a')                                  
    txt.write(str(wgt_d9L2d9[9]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9_dx2y2_dx2y2_du','a')                                  
    txt.write(str(wgt_d9L2d9[10]/number)+'\n')
    txt.close()          
    txt=open('./data/d9L2d9_dx2y2_dx2y2_dd','a')                                  
    txt.write(str(wgt_d9L2d9[11]/number)+'\n')
    txt.close()                  
    txt=open('./data/d9L2d9','a')                                  
    txt.write(str(wgt_d9L2d9[12]/number)+'\n')
    txt.close()         


    txt=open('./data/d9d9L2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d9d9L2[0]/number)+'\n')
    txt.close()        
    txt=open('./data/d9d9L2_dx2y2_d3z2r2','a')                                  
    txt.write(str(wgt_d9d9L2[1]/number)+'\n')
    txt.close()       
    txt=open('./data/d9d9L2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d9d9L2[2]/number)+'\n')
    txt.close()        
    txt=open('./data/d9d9L2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d9d9L2[3]/number)+'\n')
    txt.close()              
    txt=open('./data/d9d9L2','a')                                  
    txt.write(str(wgt_d9d9L2[4]/number)+'\n')
    txt.close()           

    txt=open('./data/d9Ld9L_b1b1','a')                                  
    txt.write(str(wgt_d9Ld9L[0]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_a1b1','a')                                  
    txt.write(str(wgt_d9Ld9L[1]/number)+'\n')
    txt.close()    
    txt=open('./data/d9Ld9L_b1a1','a')                                  
    txt.write(str(wgt_d9Ld9L[2]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_a1a1','a')                                  
    txt.write(str(wgt_d9Ld9L[3]/number)+'\n')
    txt.close()         
    txt=open('./data/d9Ld9L_b1b1s00','a')                                  
    txt.write(str(wgt_d9Ld9L[4]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_b1b1s10','a')                                  
    txt.write(str(wgt_d9Ld9L[5]/number)+'\n')
    txt.close()    
    txt=open('./data/d9Ld9L_b1b1s01','a')                                  
    txt.write(str(wgt_d9Ld9L[6]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_b1b1s11','a')                                  
    txt.write(str(wgt_d9Ld9L[7]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_a1b1s00','a')                                  
    txt.write(str(wgt_d9Ld9L[8]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_a1b1s10','a')                                  
    txt.write(str(wgt_d9Ld9L[9]/number)+'\n')
    txt.close()    
    txt=open('./data/d9Ld9L_a1b1s01','a')                                  
    txt.write(str(wgt_d9Ld9L[10]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_a1b1s11','a')                                  
    txt.write(str(wgt_d9Ld9L[11]/number)+'\n')
    txt.close()              
    txt=open('./data/d9Ld9L_b1a1s00','a')                                  
    txt.write(str(wgt_d9Ld9L[12]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_b1a1s10','a')                                  
    txt.write(str(wgt_d9Ld9L[13]/number)+'\n')
    txt.close()    
    txt=open('./data/d9Ld9L_b1a1s01','a')                                  
    txt.write(str(wgt_d9Ld9L[14]/number)+'\n')
    txt.close()      
    txt=open('./data/d9Ld9L_b1a1s11','a')                                  
    txt.write(str(wgt_d9Ld9L[15]/number)+'\n')
    txt.close()              
    txt=open('./data/d9Ld9L_a1a1s00','a')                                  
    txt.write(str(wgt_d9Ld9L[16]/number)+'\n')
    txt.close()     
    txt=open('./data/d9Ld9L_a1a1s10','a')                                  
    txt.write(str(wgt_d9Ld9L[17]/number)+'\n')
    txt.close()  
    txt=open('./data/d9Ld9L_a1a1s01','a')                                  
    txt.write(str(wgt_d9Ld9L[18]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld9L_a1a1s11','a')                                  
    txt.write(str(wgt_d9Ld9L[19]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld9L_bb_s','a')                                  
    txt.write(str(wgt_d9Ld9L[20]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld9L_bb_d','a')                                  
    txt.write(str(wgt_d9Ld9L[21]/number)+'\n')
    txt.close()              
    txt=open('./data/d9Ld9L_ab_s','a')                                  
    txt.write(str(wgt_d9Ld9L[22]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld9L_ab_d','a')                                  
    txt.write(str(wgt_d9Ld9L[23]/number)+'\n')
    txt.close()           
    txt=open('./data/d9Ld9L','a')                                  
    txt.write(str(wgt_d9Ld9L[38]/number)+'\n')
    txt.close()     

    txt=open('./data/d9d8L_dx2y2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d9d8L[0]/number)+'\n')
    txt.close()          
    txt=open('./data/d9d8L_d3z2r2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d9d8L[1]/number)+'\n')
    txt.close()              
    txt=open('./data/d9d8L_dx2y2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d9d8L[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d9d8L_d3z2r2_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d9d8L[3]/number)+'\n')
    txt.close()        
    txt=open('./data/d9d8L_dx2y2_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d9d8L[4]/number)+'\n')
    txt.close()          
    txt=open('./data/d9d8L_d3z2r2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d9d8L[5]/number)+'\n')
    txt.close()  
    txt=open('./data/d9d8L_dx2y2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d9d8L[6]/number)+'\n')
    txt.close()  
    txt=open('./data/d9d8L','a')                                  
    txt.write(str(wgt_d9d8L[7]/number)+'\n')
    txt.close()   
    txt=open('./data/d9d8L_singlet','a')                                  
    txt.write(str((wgt_d9d8L[7]/number-wgt_d9d8L[3]/number-wgt_d9d8L[4]/number)/number)+'\n')
    txt.close()    
    txt=open('./data/d9d8L_triplet','a')                                  
    txt.write(str((wgt_d9d8L[3]/number+wgt_d9d8L[4]/number)/number)+'\n')
    txt.close()            

    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2','a')                                  
    txt.write(str(wgt_d8d9L[0]/number)+'\n')
    txt.close()            
    txt=open('./data/d8d9L_dx2y2_dx2y2_d3z2r2','a')                                  
    txt.write(str(wgt_d8d9L[1]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d8d9L[2]/number)+'\n')
    txt.close()        
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8d9L[3]/number)+'\n')
    txt.close()  
    txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8d9L[4]/number)+'\n')
    txt.close()  
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_00','a')                                  
    txt.write(str(wgt_d8d9L[5]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_10','a')                                  
    txt.write(str(wgt_d8d9L[6]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_01','a')                                  
    txt.write(str(wgt_d8d9L[7]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_11','a')                                  
    txt.write(str(wgt_d8d9L[8]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_dx2y2_dx2y2_d3z2r2_00','a')                                  
    txt.write(str(wgt_d8d9L[9]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_dx2y2_dx2y2_d3z2r2_10','a')                                  
    txt.write(str(wgt_d8d9L[10]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_dx2y2_dx2y2_d3z2r2_01','a')                                  
    txt.write(str(wgt_d8d9L[11]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_dx2y2_dx2y2_d3z2r2_11','a')                                  
    txt.write(str(wgt_d8d9L[12]/number)+'\n')
    txt.close()            
    txt=open('./data/d8d9L_d3z2r2_d3z2r2_dx2y2_00','a')                                  
    txt.write(str(wgt_d8d9L[13]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_d3z2r2_dx2y2_10','a')                                  
    txt.write(str(wgt_d8d9L[14]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_d3z2r2_d3z2r2_dx2y2_01','a')                                  
    txt.write(str(wgt_d8d9L[15]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_d3z2r2_d3z2r2_dx2y2_11','a')                                  
    txt.write(str(wgt_d8d9L[16]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_00','a')                                  
    txt.write(str(wgt_d8d9L[17]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_10','a')                                  
    txt.write(str(wgt_d8d9L[18]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_01','a')                                  
    txt.write(str(wgt_d8d9L[19]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_11','a')                                  
    txt.write(str(wgt_d8d9L[20]/number)+'\n')
    txt.close()    
    txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2_00','a')                                  
    txt.write(str(wgt_d8d9L[21]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2_10','a')                                  
    txt.write(str(wgt_d8d9L[22]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2_01','a')                                  
    txt.write(str(wgt_d8d9L[23]/number)+'\n')
    txt.close()               
    txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2_11','a')                                  
    txt.write(str(wgt_d8d9L[24]/number)+'\n')
    txt.close()      
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_11ss','a')                                  
    txt.write(str(wgt_d8d9L[25]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_11sd','a')                                  
    txt.write(str(wgt_d8d9L[26]/number)+'\n')
    txt.close()                  
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_11ds','a')                                  
    txt.write(str(wgt_d8d9L[27]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2_11dd','a')                                  
    txt.write(str(wgt_d8d9L[28]/number)+'\n')
    txt.close()                   
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_10s','a')                                  
    txt.write(str(wgt_d8d9L[29]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_dx2y2_dx2y2_10d','a')                                  
    txt.write(str(wgt_d8d9L[30]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d9L','a')                                  
    txt.write(str(wgt_d8d9L[31]/number)+'\n')
    txt.close()    
    txt=open('./data/d8d9L+d9Ld8','a')                                  
    txt.write(str(wgt_d8d9L[31]/number+wgt_d9Ld8[8]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d9L_dx2y2_dx2y2_dx2y2+d9Ld8','a')                                  
    txt.write(str(wgt_d8d9L[4]/number+wgt_d9Ld8[6]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9L_d3z2r2_dx2y2_d3z2r2+d9Ld8','a')                                  
    txt.write(str(wgt_d8d9L[0]/number+wgt_d9Ld8[1]/number)+'\n')
    txt.close()             


    txt=open('./data/d9Ld8_singlet','a')                                  
    txt.write(str(wgt_d8d9L[5]/number+wgt_d8d9L[7]/number+wgt_d8d9L[9]/number+wgt_d8d9L[11]/number+wgt_d8d9L[13]/number+wgt_d8d9L[15]/number \
              +wgt_d8d9L[17]/number+wgt_d8d9L[19]/number+wgt_d8d9L[21]/number+wgt_d8d9L[23]/number)+'\n')
    txt=open('./data/d9Ld8_triplet','a')
    txt.write(str(wgt_d8d9L[6]/number+wgt_d8d9L[8]/number+wgt_d8d9L[10]/number+wgt_d8d9L[12]/number+wgt_d8d9L[14]/number+wgt_d8d9L[16]/number \
              +wgt_d8d9L[18]/number+wgt_d8d9L[20]/number+wgt_d8d9L[22]/number+wgt_d8d9L[24]/number)+'\n')
    txt.close()


    txt=open('./data/d9Ld8_dx2y2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d9Ld8[0]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld8_d3z2r2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d9Ld8[1]/number)+'\n')
    txt.close()              
    txt=open('./data/d9Ld8_dx2y2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d9Ld8[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld8_d3z2r2_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d9Ld8[3]/number)+'\n')
    txt.close()        
    txt=open('./data/d9Ld8_dx2y2_d3z2r2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d9Ld8[4]/number)+'\n')
    txt.close()          
    txt=open('./data/d9Ld8_d3z2r2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d9Ld8[5]/number)+'\n')
    txt.close() 
    txt=open('./data/d9Ld8_dx2y2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d9Ld8[6]/number)+'\n')
    txt.close() 
    txt=open('./data/d9Ld8','a')                                  
    txt.write(str(wgt_d9Ld8[8]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d9L_singlet','a')                                  
    txt.write(str((wgt_d9Ld8[7]/number-wgt_d9Ld8[3]/number-wgt_d9Ld8[4]/number))+'\n')
    txt.close()            
    txt=open('./data/d8d9L_triplet','a')                                  
    txt.write(str((wgt_d9Ld8[3]/number+wgt_d9Ld8[4]/number))+'\n')
    txt.close()          



    txt=open('./data/d8Ld9_d3z2r2_dx2y2_d3z2r2','a')                                  
    txt.write(str(wgt_d8Ld9[0]/number)+'\n')
    txt.close()          
    txt=open('./data/d8Ld9_d3z2r2_dx2y2_d3z2r2_S1','a')                                  
    txt.write(str(wgt_d8Ld9[1]/number)+'\n')
    txt.close()              
    txt=open('./data/d8Ld9_dx2y2_dx2y2_d3z2r2','a')                                  
    txt.write(str(wgt_d8Ld9[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d8Ld9_d3z2r2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d8Ld9[3]/number)+'\n')
    txt.close()        
    txt=open('./data/d8Ld9_d3z2r2_dx2y2_dx2y2_S1','a')                                  
    txt.write(str(wgt_d8Ld9[4]/number)+'\n')
    txt.close()          
    txt=open('./data/d8Ld9_d3z2r2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8Ld9[5]/number)+'\n')
    txt.close() 
    txt=open('./data/d8Ld9_dx2y2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8Ld9[6]/number)+'\n')
    txt.close() 
    txt=open('./data/d8Ld9','a')                                  
    txt.write(str(wgt_d8Ld9[7]/number)+'\n')
    txt.close()    
            



    txt=open('./data/d8d8_dx2y2_dx2y2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d8d8[0]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2','a')                                  
    txt.write(str(wgt_d8d8[1]/number)+'\n')
    txt.close()              
    txt=open('./data/d8d8_d3z2r2_d3z2r2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8d8[2]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d8_dx2y2_dx2y2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8d8[3]/number)+'\n')
    txt.close()      
    txt=open('./data/d8d8_d3z2r2_dx2y2_dx2y2_dx2y2','a')                                  
    txt.write(str(wgt_d8d8[4]/number)+'\n')
    txt.close()        
    txt=open('./data/d8d8_d3z2r2_d3z2r2_d3z2r2_d3z2r2','a')                                  
    txt.write(str(wgt_d8d8[5]/number)+'\n')
    txt.close()      
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_00','a')                                  
    txt.write(str(wgt_d8d8[6]/number)+'\n')
    txt.close()           
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_10','a')                                  
    txt.write(str(wgt_d8d8[7]/number)+'\n')
    txt.close()                   
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_01','a')                                  
    txt.write(str(wgt_d8d8[8]/number)+'\n')
    txt.close()                   
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_11','a')                                  
    txt.write(str(wgt_d8d8[9]/number)+'\n')
    txt.close()                   
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_11ss','a')                                  
    txt.write(str(wgt_d8d8[10]/number)+'\n')
    txt.close()  
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_11sd','a')                                  
    txt.write(str(wgt_d8d8[11]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_11ds','a')                                  
    txt.write(str(wgt_d8d8[12]/number)+'\n')
    txt.close()  
    txt=open('./data/d8d8_d3z2r2_dx2y2_d3z2r2_dx2y2_11dd','a')                                  
    txt.write(str(wgt_d8d8[13]/number)+'\n')
    txt.close()                  


    txt=open('./data/d8d8','a')                                  
    txt.write(str(wgt_d8d8[14]/number)+'\n')
    txt.close()     


    txt=open('./data/s','a')                                  
    txt.write(str(wgt_s[0]/number)+'\n')
    txt.close()     

    txt=open('./data/ddss','a')                                  
    txt.write(str(wgt_ddss[0]/number)+'\n')
    txt.close()        
    txt=open('./data/ddss_11','a')                                  
    txt.write(str(wgt_ddss[1]/number)+'\n')
    txt.close()            
    txt=open('./data/ddss_00','a')                                  
    txt.write(str(wgt_ddss[2]/number)+'\n')
    txt.close()      
    txt=open('./data/ddss_10','a')                                  
    txt.write(str(wgt_ddss[3]/number)+'\n')
    txt.close()      
    txt=open('./data/ddss_11_a1a1','a')                                  
    txt.write(str(wgt_ddss[4]/number)+'\n')
    txt.close()     
    txt=open('./data/ddss_11_a1b1','a')                                  
    txt.write(str(wgt_ddss[5]/number)+'\n')
    txt.close()        
    txt=open('./data/ddss_11_b1b1','a')                                  
    txt.write(str(wgt_ddss[6]/number)+'\n')
    txt.close()            
    txt=open('./data/ddss_00_a1a1','a')                                  
    txt.write(str(wgt_ddss[7]/number)+'\n')
    txt.close()     
    txt=open('./data/ddss_00_a1b1','a')                                  
    txt.write(str(wgt_ddss[8]/number)+'\n')
    txt.close()        
    txt=open('./data/ddss_00_b1b1','a')                                  
    txt.write(str(wgt_ddss[9]/number)+'\n')
    txt.close()               
    txt=open('./data/ddss_10_a1a1','a')                                  
    txt.write(str(wgt_ddss[10]/number)+'\n')
    txt.close()     
    txt=open('./data/ddss_10_a1b1','a')                                  
    txt.write(str(wgt_ddss[11]/number)+'\n')
    txt.close()        
    txt=open('./data/ddss_10_b1b1','a')                                  
    txt.write(str(wgt_ddss[12]/number)+'\n')
    txt.close()    


    txt=open('./data/ddsp','a')                                  
    txt.write(str(wgt_ddsp[0]/number)+'\n')
    txt.close()        
    txt=open('./data/ddsp_11','a')                                  
    txt.write(str(wgt_ddsp[1]/number)+'\n')
    txt.close()            
    txt=open('./data/ddsp_00','a')                                  
    txt.write(str(wgt_ddsp[2]/number)+'\n')
    txt.close()      
    txt=open('./data/ddsp_10','a')                                  
    txt.write(str(wgt_ddsp[3]/number)+'\n')
    txt.close()      
    txt=open('./data/ddsp_11_a1a1','a')                                  
    txt.write(str(wgt_ddsp[4]/number)+'\n')
    txt.close()     
    txt=open('./data/ddsp_11_a1b1','a')                                  
    txt.write(str(wgt_ddsp[5]/number)+'\n')
    txt.close()        
    txt=open('./data/ddsp_11_b1b1','a')                                  
    txt.write(str(wgt_ddsp[6]/number)+'\n')
    txt.close()            
    txt=open('./data/ddsp_00_a1a1','a')                                  
    txt.write(str(wgt_ddsp[7]/number)+'\n')
    txt.close()     
    txt=open('./data/ddsp_00_a1b1','a')                                  
    txt.write(str(wgt_ddsp[8]/number)+'\n')
    txt.close()        
    txt=open('./data/ddsp_00_b1b1','a')                                  
    txt.write(str(wgt_ddsp[9]/number)+'\n')
    txt.close()               
    txt=open('./data/ddsp_10_a1a1','a')                                  
    txt.write(str(wgt_ddsp[10]/number)+'\n')
    txt.close()     
    txt=open('./data/ddsp_10_a1b1','a')                                  
    txt.write(str(wgt_ddsp[11]/number)+'\n')
    txt.close()        
    txt=open('./data/ddsp_10_b1b1','a')                                  
    txt.write(str(wgt_ddsp[12]/number)+'\n')
    txt.close()    
    txt=open('./data/ddsp_10_a1b1_z2','a')                                  
    txt.write(str(wgt_ddsp[13]/number)+'\n')
    txt.close()            
    txt=open('./data/ddsp_10_a1b1_z0','a')                                  
    txt.write(str(wgt_ddsp[14]/number)+'\n')
    txt.close()            
    txt=open('./data/ddsp_10_a1b1_z2_1','a')                                  
    txt.write(str(wgt_ddsp[15]/number)+'\n')
    txt.close()            
    txt=open('./data/ddsp_10_a1b1_z2_0','a')                                  
    txt.write(str(wgt_ddsp[16]/number)+'\n')
    txt.close()     
    txt=open('./data/ddsp_10_1','a')                                  
    txt.write(str(wgt_ddsp[17]/number)+'\n')
    txt.close()             
    txt=open('./data/ddsp_10_0','a')                                  
    txt.write(str(wgt_ddsp[18]/number)+'\n')
    txt.close()            

    txt=open('./data/dssp','a')                                  
    txt.write(str(wgt_dssp[0]/number)+'\n')
    txt.close()     
    txt=open('./data/dssp_22','a')                                  
    txt.write(str(wgt_dssp[1]/number)+'\n')
    txt.close()        
    txt=open('./data/dssp_20','a')                                  
    txt.write(str(wgt_dssp[2]/number)+'\n')
    txt.close()     
    txt=open('./data/dssp_22_a1','a')                                  
    txt.write(str(wgt_dssp[3]/number)+'\n')
    txt.close()               
    txt=open('./data/dssp_22_b1','a')                                  
    txt.write(str(wgt_dssp[4]/number)+'\n')
    txt.close()                   
    txt=open('./data/dssp_20_a1','a')                                  
    txt.write(str(wgt_dssp[5]/number)+'\n')
    txt.close()               
    txt=open('./data/dssp_20_b1','a')                                  
    txt.write(str(wgt_dssp[6]/number)+'\n')
    txt.close()          

    txt=open('./data/dspp','a')                                  
    txt.write(str(wgt_dspp[0]/number)+'\n')
    txt.close()     
    txt=open('./data/dspp_10','a')                                  
    txt.write(str(wgt_dspp[1]/number)+'\n')
    txt.close()        
    txt=open('./data/dspp_00','a')                                  
    txt.write(str(wgt_dspp[2]/number)+'\n')
    txt.close()     
    txt=open('./data/dspp_1_a1','a')                                  
    txt.write(str(wgt_dspp[3]/number)+'\n')
    txt.close()               
    txt=open('./data/dspp_1_b1','a')                                  
    txt.write(str(wgt_dspp[4]/number)+'\n')
    txt.close()                   
    txt=open('./data/dspp_0_a1','a')                                  
    txt.write(str(wgt_dspp[5]/number)+'\n')
    txt.close()               
    txt=open('./data/dspp_0_b1','a')                                  
    txt.write(str(wgt_dspp[6]/number)+'\n')
    txt.close()     

    txt=open('./data/ddds','a')                                  
    txt.write(str(wgt_ddds[0]/number)+'\n')
    txt.close()            
    txt=open('./data/ddds_110','a')                                  
    txt.write(str(wgt_ddds[1]/number)+'\n')
    txt.close()             
    txt=open('./data/ddds_100','a')                                  
    txt.write(str(wgt_ddds[2]/number)+'\n')
    txt.close()              
    txt=open('./data/ddds_110_a1a1a1','a')                                  
    txt.write(str(wgt_ddds[3]/number)+'\n')
    txt.close()             
    txt=open('./data/ddds_110_a1b1a1','a')                                  
    txt.write(str(wgt_ddds[4]/number)+'\n')
    txt.close()                
    txt=open('./data/ddds_110_b1b1a1','a')                                  
    txt.write(str(wgt_ddds[5]/number)+'\n')
    txt.close()                     
    txt=open('./data/ddds_110_a1a1b1','a')                                  
    txt.write(str(wgt_ddds[6]/number)+'\n')
    txt.close()             
    txt=open('./data/ddds_110_a1b1b1','a')                                  
    txt.write(str(wgt_ddds[7]/number)+'\n')
    txt.close()                
    txt=open('./data/ddds_110_b1b1b1','a')                                  
    txt.write(str(wgt_ddds[8]/number)+'\n')
    txt.close()          
    txt=open('./data/ddds_100_a1a1a1','a')                                  
    txt.write(str(wgt_ddds[9]/number)+'\n')
    txt.close()             
    txt=open('./data/ddds_100_a1a1b1','a')                                  
    txt.write(str(wgt_ddds[10]/number)+'\n')
    txt.close()        
    txt=open('./data/ddds_100_a1b1b1','a')                                  
    txt.write(str(wgt_ddds[11]/number)+'\n')
    txt.close()                
    txt=open('./data/ddds_100_b1a1a1','a')                                  
    txt.write(str(wgt_ddds[12]/number)+'\n')
    txt.close()             
    txt=open('./data/ddds_100_b1a1b1','a')                                  
    txt.write(str(wgt_ddds[13]/number)+'\n')
    txt.close()        
    txt=open('./data/ddds_100_b1b1b1','a')                                  
    txt.write(str(wgt_ddds[14]/number)+'\n')
    txt.close()                   
    txt=open('./data/ddds_110_a1b1b1_1','a')                                  
    txt.write(str(wgt_ddds[15]/number)+'\n')
    txt.close()        
    txt=open('./data/ddds_110_a1b1b1_0','a')                                  
    txt.write(str(wgt_ddds[16]/number)+'\n')
    txt.close()        


    txt=open('./data/number','a')                                  
    txt.write(str(number)+'\n')
    txt.close() 

    print ('sumweight2=',sumweight2)        
    print ('sumweight_picture=',sumweight_picture)
    print ('LmLn=',wgt_LmLn[0]/number,'  bonding0=',wgt_LmLn[46]/number,'  bonding1=',wgt_LmLn[47]/number,'  bonding-1=',wgt_LmLn[48]/number)
    print ('d9Ld10L2=',wgt_d9Ld10L2[12]/number+wgt_d10L2d9L[2]/number,'  bonding0=',wgt_d9Ld10L2[46]/number+wgt_d10L2d9L[46]/number,'  bonding1=',wgt_d9Ld10L2[47]/number+wgt_d10L2d9L[47]/number,'  bonding-1=',wgt_d9Ld10L2[48]/number+wgt_d10L2d9L[48]/number)
    print ('d9d10L3=',wgt_d9d10L3[2]/number+wgt_d10L3d9[10]/number,'  bonding0=',wgt_d9d10L3[46]/number+wgt_d10L3d9[46]/number,'  bonding1=',wgt_d9d10L3[47]/number+wgt_d10L3d9[47]/number,'  bonding-1=',wgt_d9d10L3[48]/number+wgt_d10L3d9[48]/number)        
            
    print ('d9L2d10L=',wgt_d9L2d10L[2]/number+wgt_d10Ld9L2[2]/number,'  bonding0=',wgt_d9L2d10L[46]/number+wgt_d10Ld9L2[46]/number,'  bonding1=',wgt_d9L2d10L[47]/number+wgt_d10Ld9L2[47]/number,'  bonding-1=',wgt_d9L2d10L[48]/number+wgt_d10Ld9L2[48]/number)
     
#     print ('d10d9L3=',wgt_d10d9L3[2]/number)

    print ('d8Ld10L=',wgt_d8Ld10L[4]/number+wgt_d10Ld8L[4]/number,'  bonding0=',wgt_d8Ld10L[46]/number+wgt_d10Ld8L[46]/number,'  bonding1=',wgt_d8Ld10L[47]/number+wgt_d10Ld8L[47]/number,'  bonding-1=',wgt_d8Ld10L[48]/number+wgt_d10Ld8L[48]/number)
    
    print ('d8d10L2=',wgt_d8d10L2[4]/number+wgt_d10L2d8[4]/number,'  bonding0=',wgt_d8d10L2[46]/number+wgt_d10L2d8[46]/number,'  bonding1=',wgt_d8d10L2[47]/number+wgt_d10L2d8[47]/number,'  bonding-1=',wgt_d8d10L2[48]/number+wgt_d10L2d8[48]/number)
    print ('d10d8L2=',wgt_d10d8L2[4]/number+wgt_d8L2d10[4]/number,'  bonding0=',wgt_d10d8L2[46]/number+wgt_d8L2d10[46]/number,'  bonding1=',wgt_d10d8L2[47]/number+wgt_d8L2d10[47]/number,'  bonding-1=',wgt_d10d8L2[48]/number+wgt_d8L2d10[48]/number) 

    print ('d9L2d9=',wgt_d9L2d9[12]/number+wgt_d9d9L2[4]/number,'  bonding0=',wgt_d9L2d9[46]/number+wgt_d9d9L2[46]/number,'  bonding1=',wgt_d9L2d9[47]/number+wgt_d9d9L2[47]/number,'  bonding-1=',wgt_d9L2d9[48]/number+wgt_d9d9L2[48]/number)
      
    print ('d9Ld9L=',wgt_d9Ld9L[39]/number,'  bonding0=',wgt_d9Ld9L[46]/number,'  bonding1=',wgt_d9Ld9L[47]/number,'  bonding-1=',wgt_d9Ld9L[48]/number)        
    print ('d9d8L=',wgt_d9d8L[7]/number+wgt_d8Ld9[7]/number,'  bonding0=',wgt_d9d8L[46]/number+wgt_d8Ld9[46]/number,'  bonding1=',wgt_d9d8L[47]/number+wgt_d8Ld9[47]/number,'  bonding-1=',wgt_d9d8L[48]/number+wgt_d8Ld9[48]/number)        
    print ('d8d9L=',wgt_d8d9L[31]/number+wgt_d9Ld8[8]/number,'  bonding0=',wgt_d8d9L[46]/number+wgt_d9Ld8[46]/number,'  bonding1=',wgt_d8d9L[47]/number+wgt_d9Ld8[47]/number,'  bonding-1=',wgt_d8d9L[48]/number+wgt_d9Ld8[48]/number)
       
    print ('d8d8=',wgt_d8d8[14]/number,'  bonding0=',wgt_d8d8[46]/number,'  bonding1=',wgt_d8d8[47]/number,'  bonding-1=',wgt_d8d8[48]/number)

    print ('text0=',text[0]/number)
    print ('text1=',text[1]/number)
    print ('text-1=',text[2]/number)    
    print("--- get_ground_state %s seconds ---" % (time.time() - t1))
                
    return vals, vecs 

#########################################################################
    # set up Lanczos solver
#     dim  = VS.dim
#     scratch = np.empty(dim, dtype = complex)
    
#     #`x0`: Starting vector. Use something randomly initialized
#     Phi0 = np.zeros(dim, dtype = complex)
#     Phi0[10] = 1.0
    
#     vecs = np.zeros(dim, dtype = complex)
#     solver = lanczos.LanczosSolver(maxiter = 200, 
#                                    precision = 1e-12, 
#                                    cond = 'UPTOMAX', 
#                                    eps = 1e-8)
#     vals = solver.lanczos(x0=Phi0, scratch=scratch, y=vecs, H=matrix)
#     print ('GS energy = ', vals)
    
#     # get state components in GS; note that indices is a tuple
#     indices = np.nonzero(abs(vecs)>0.01)
#     wgt_d8 = np.zeros(6)
#     wgt_d9L = np.zeros(4)
#     wgt_d10L2 = np.zeros(1)

#     print ("Compute the weights in GS (lowest Aw peak)")
#     #for i in indices[0]:
#     for i in range(0,len(vecs)):
#         # state is original state but its orbital info remains after basis change
#         state = VS.get_state(VS.lookup_tbl[i])
 
#         s1 = state['hole1_spin']
#         s2 = state['hole2_spin']
#         s3 = state['hole3_spin']
#         orb1 = state['hole1_orb']
#         orb2 = state['hole2_orb']
#         orb3 = state['hole3_orb']
#         x1, y1, z1 = state['hole1_coord']
#         x2, y2, z2 = state['hole2_coord']
#         x3, y3, z3 = state['hole3_coord']

#         #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
#         #    continue
#         S12  = S_val[i]
#         Sz12 = Sz_val[i]

#         o12 = sorted([orb1,orb2,orb3])
#         o12 = tuple(o12)

#         if i in indices[0]:
#             print (' state ', orb1,s1,x1,y1,z1,orb2,s2,x2,y2,z2,orb3,s3,x3,y3,z3 ,'S=',S12,'Sz=',Sz12,", weight = ", abs(vecs[i,k])**2)
#     return vals, vecs, wgt_d8, wgt_d9L, wgt_d10L2
