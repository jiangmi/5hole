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
    s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];    
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
            elif  tmp24 != tlabel:
                slabel2 = [s4,orb4,x4,y4,z4] + slabel[0:15]     
                
    tlabel = slabel2[15:20] + [s5,orb5,x5,y5,z5]
    tmp45 = reorder_z(tlabel)                
    if tmp45 == tlabel:
        slabel3 = slabel2 + [s5,orb5,x5,y5,z5]
    else:
        tlabel = slabel2[10:15] + [s5,orb5,x5,y5,z5] 
        tmp35 = reorder_z(tlabel)                         
        if tmp35 == tlabel:
            slabel3 = slabel2[0:15] + [s5,orb5,x5,y5,z5] + slabel2[15:20]
        else:
            tlabel = slabel2[5:10] + [s5,orb5,x5,y5,z5] 
            tmp25 = reorder_z(tlabel)                          
            if tmp25 == tlabel:
                slabel3 = slabel2[0:10] + [s5,orb5,x5,y5,z5] + slabel2[10:20]   
            else:
                tlabel = slabel2[0:5] + [s5,orb5,x5,y5,z5] 
                tmp15 = reorder_z(tlabel)                          
                if tmp25 == tlabel:
                    slabel3 = slabel2[0:5] + [s5,orb5,x5,y5,z5] + slabel2[5:20]     
                else:
                    slabel3 = [s5,orb5,x5,y5,z5] + slabel2                       
                        
                
                
    return slabel3


def get_ground_state(matrix, VS, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val, tz):  
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
    
    #get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,number):                                                                          #gai
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.1)

        wgt_LmLn = np.zeros(10)
        wgt_d8d8L = np.zeros(20)
        wgt_d8Ld8 = np.zeros(10)
        wgt_d9Ld8L = np.zeros(20)        
        wgt_d9L2d8= np.zeros(10)
        wgt_d9d8L2= np.zeros(10)
        wgt_d9L3d9= np.zeros(10)   
        wgt_d9L2d9L= np.zeros(10)  
        wgt_d9Ld9L2= np.zeros(10)
        wgt_d9d9L3= np.zeros(10)        
        wgt_d8L3d10= np.zeros(10)   
        wgt_d8L2d10L= np.zeros(10)  
        wgt_d8Ld10L2= np.zeros(10)
        wgt_d8d10L3= np.zeros(10) 
        wgt_d9d10L4= np.zeros(10)
        wgt_d9Ld10L3= np.zeros(10)        
        wgt_d9L2d10L2= np.zeros(10)          
        wgt_d9L3d10L= np.zeros(10)   
        wgt_d9L4d10= np.zeros(10)           
        
        
        
        
        
#         s11=0
#         s10=0        
#         s01=0
#         s00=0        
        #Sumweight refers to the general weight.Sumweight1 refers to the weight in indices.Sumweight_picture refers to the weight that is calculated.Sumweight2 refers to the weight that differs by orbits

        sumweight=0
        sumweight1=0
        synweight2=0

        print ("Compute the weights in GS (lowest Aw peak)")
        #for i in indices[0]:
        for i in range(0,len(vecs[:,k])):
            # state is original state but its orbital info remains after basis change
            state = VS.get_state(VS.lookup_tbl[i])
            weight = abs(vecs[i,k])**2
            
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']
            s4 = state['hole4_spin'] 
            s5 = state['hole5_spin']             
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']
            orb4 = state['hole4_orb'] 
            orb5 = state['hole5_orb']             
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']
            x4, y4, z4 = state['hole4_coord']  
            x5, y5, z5 = state['hole5_coord']              

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S_Ni_12  = S_Ni_val[i]
            Sz_Ni_12 = Sz_Ni_val[i]
            S_Cu_12  = S_Cu_val[i]
            Sz_Cu_12 = Sz_Cu_val[i]
            
#             S_Niother_12  = S_other_Ni_val[i]
#             Sz_Niother_12 = Sz_other_Ni_val[i]
#             S_Cuother_12  = S_other_Cu_val[i]
#             Sz_Cuother_12 = Sz_other_Cu_val[i]
                
            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,s5,orb5,x5,y5,z5]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
            s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];     
            s5 = slabel[20]; orb5 = slabel[21]; x5 = slabel[22]; y5 = slabel[23]; z5 = slabel[24];                
    
            if i in indices[0]: 
                sumweight1=sumweight1+abs(vecs[i,k])**2
                print (' state ', i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,' ',orb5,s5,x5,y5,z5,\
                   '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
                   ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
                   ", weight = ", weight,'\n')   
                
                
            if (orb1 in pam.O_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs)  and  (orb4 in pam.O_orbs) and  (orb5 in pam.O_orbs): 
                wgt_LmLn[0]+=abs(vecs[i,k])**2 
  

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and z1==z2==z5==1 and z3==z4==0 : 
                wgt_d8Ld8[0]+=abs(vecs[i,k])**2 


            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.Ni_Cu_orbs) and (orb5 in pam.O_orbs) and z1==z2==1 and z3==z4==z5==0 : 
                wgt_d8d8L[0]+=abs(vecs[i,k])**2             
              
            

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z4==1 and z2==z3==z5==0 : 
                wgt_d9Ld8L[0]+=abs(vecs[i,k])**2               
            
            
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z4==z5==1 and z2==z3==0 : 
                wgt_d9L2d8[0]+=abs(vecs[i,k])**2 
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.Ni_Cu_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==1 and z2==z3==z4==z5==0 : 
                wgt_d9d8L2[0]+=abs(vecs[i,k])**2                 

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==1 and z2==z3==z4==z5==0 : 
                wgt_d9d9L3[0]+=abs(vecs[i,k])**2  
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==1 and z2==z4==z5==0 : 
                wgt_d9Ld9L2[0]+=abs(vecs[i,k])**2                  
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==z4==1 and z2==z5==0 : 
                wgt_d9L2d9L[0]+=abs(vecs[i,k])**2                   
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z3==z4==z5==1 and z2==0 : 
                wgt_d9L3d9[0]+=abs(vecs[i,k])**2                  

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==1 and z3==z4==z5==0 : 
                wgt_d8d10L3[0]+=abs(vecs[i,k])**2  
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==1 and z4==z5==0 : 
                wgt_d8Ld10L2[0]+=abs(vecs[i,k])**2                  
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==1 and z5==0 : 
                wgt_d8L2d10L[0]+=abs(vecs[i,k])**2                   
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==z5==1 : 
                wgt_d8L3d10[0]+=abs(vecs[i,k])**2                  
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==1 and z2==z3==z4==z5==0 : 
                wgt_d9d10L4[0]+=abs(vecs[i,k])**2                     
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==1 and z3==z4==z5==0 : 
                wgt_d9Ld10L3[0]+=abs(vecs[i,k])**2                        
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==1 and z4==z5==0 : 
                wgt_d9L2d10L2[0]+=abs(vecs[i,k])**2                        
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==1 and z5==0 : 
                wgt_d9L3d10L[0]+=abs(vecs[i,k])**2    
                
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and  (orb3 in pam.O_orbs) and (orb4 in pam.O_orbs) and (orb5 in pam.O_orbs) and z1==z2==z3==z4==z5==1: 
                wgt_d9L4d10[0]+=abs(vecs[i,k])**2                   
                
                
            sumweight=sumweight+abs(vecs[i,k])**2
        print ('sumweight=',sumweight)
        print ('LmLn=',wgt_LmLn[0])
        print ('d8Ld8=',wgt_d8Ld8[0])
        print ('d8d8L=',wgt_d8d8L[0])       
        print ('d9Ld8L=',wgt_d9Ld8L[0])             
        print ('d9d8L2=',wgt_d9d8L2[0])          
        print ('d9L2d8=',wgt_d9L2d8[0])         
        print ('d9L3d9=',wgt_d9L3d9[0]) 
        print ('d9L2d9L=',wgt_d9L2d9L[0])         
        print ('d9Ld9L2=',wgt_d9Ld9L2[0])         
        print ('d9d9L3=',wgt_d9d9L3[0])         
        print ('d8L3d10=',wgt_d8L3d10[0]) 
        print ('d8L2d10L=',wgt_d8L2d10L[0])         
        print ('d8Ld10L2=',wgt_d8Ld10L2[0])         
        print ('d8d10L3=',wgt_d8d10L3[0])  
        print ('d9d10L4=',wgt_d9d10L4[0])          
        print ('d9Ld10L3=',wgt_d9Ld10L3[0])         
        print ('d9L2d10L2=',wgt_d9L2d10L2[0])         
        print ('d9L3d10L=',wgt_d9L3d10L[0]) 
        print ('d9L4d10=',wgt_d9L4d10[0])         

        sumweight2 = wgt_LmLn[0]+wgt_d8Ld8[0]+wgt_d8d8L[0]+wgt_d9Ld8L[0]+wgt_d9d8L2[0]+wgt_d9L2d8[0]+wgt_d9L3d9[0]+wgt_d9L2d9L[0]\
                 +wgt_d9Ld9L2[0]+wgt_d9d9L3[0]+wgt_d8L3d10[0]+wgt_d8L2d10L[0]+wgt_d8Ld10L2[0]+wgt_d8d10L3[0]+wgt_d9d10L4[0]\
                 +wgt_d9Ld10L3[0]+wgt_d9L2d10L2[0]+wgt_d9L3d10L[0]+wgt_d9L4d10[0]
        print ('sumweight2=',sumweight2)
#         print ('s11=',s11)        
#         print ('s10=',s10)       
#         print ('s01=',s01)  
#         print ('s00=',s00)          







        path = './data'		# create file

        if os.path.isdir(path) == False:
            os.mkdir(path) 

#         txt=open('./data/LmLn','a')                                  
#         txt.write(str(wgt_LmLn[0])+'\n')
#         txt.close()     
#         txt=open('./data/LmLn_00','a')                                  
#         txt.write(str(wgt_LmLn[1])+'\n')
#         txt.close()   
#         txt=open('./data/LmLn_10','a')                                  
#         txt.write(str(wgt_LmLn[2])+'\n')
#         txt.close()           
#         txt=open('./data/LmLn_01','a')                                  
#         txt.write(str(wgt_LmLn[3])+'\n')
#         txt.close()           
#         txt=open('./data/LmLn_11','a')                                  
#         txt.write(str(wgt_LmLn[4])+'\n')
#         txt.close()           


     

        txt=open('./data/number','a')                                  
        txt.write(str(number)+'\n')
        txt.close() 




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
