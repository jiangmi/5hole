import variational_space as vs
import lattice as lat
import bisect
import numpy as np
import scipy.sparse as sps
import parameters as pam
import utility as util
                
def find_singlet_triplet_partner_d_double(VS, d_part, index, h34_part):
    '''
    For a given state find its partner state to form a singlet/triplet.
    Right now only applied for d_double states
    
    Note: idx is to label which hole is not on Ni

    Returns
    -------
    index: index of the singlet/triplet partner state in the VS
    phase: phase factor with which the partner state needs to be multiplied.
    '''
    if index==14:
        slabel = h34_part[0:5] + [d_part[0]]+d_part[6:10] + [d_part[5]]+d_part[1:5] +h34_part[5:10]
    elif index==24:
        slabel = [d_part[0]]+d_part[6:10] + h34_part[0:5] + [d_part[5]]+d_part[1:5] +h34_part[5:10]
    elif index==34:
        slabel = [d_part[0]]+d_part[6:10] + [d_part[5]]+d_part[1:5] + h34_part[0:5] +h34_part[5:10]
    elif index==13:
        slabel = h34_part[0:5] + [d_part[0]]+d_part[6:10]+h34_part[5:10]  + [d_part[5]]+d_part[1:5]
    elif index==23:
        slabel = [d_part[0]]+d_part[6:10] + h34_part[0:5] +h34_part[5:10]  +[d_part[5]]+d_part[1:5]
    elif index==12:
        slabel = h34_part[0:5] +h34_part[5:10] +[d_part[0]]+d_part[6:10] +[d_part[5]]+d_part[1:5]         
                        
    tmp_state = vs.create_state(slabel)
    partner_state,_,_ = vs.make_state_canonical(tmp_state)
    phase = -1.0
 
    return VS.get_index(partner_state), phase


def create_singlet_triplet_basis_change_matrix_d_double(VS, d_double, double_part, idx, hole34_part):
    '''
    Similar to above create_singlet_triplet_basis_change_matrix but only applies
    basis change for d_double states
    
    Note that for three hole state, its partner state must have exactly the same
    spin and positions of L and Nd-electron
    
    This function is required for create_interaction_matrix_ALL_syms !!!
    '''
    data = []
    row = []
    col = []
    
    count_singlet = 0
    count_triplet = 0
    
    # store index of partner state in d_double to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_d8_val  = np.zeros(VS.dim, dtype=int)
    Sz_d8_val = np.zeros(VS.dim, dtype=int)
    AorB_d8_sym = np.zeros(VS.dim, dtype=int)
    
    # first set the matrix to be identity matrix (for states not d_double)

    for i in range(VS.dim):
        if i not in d_double:
            data.append(np.sqrt(2.0)); row.append(i); col.append(i)
        
    for i, double_id in enumerate(d_double):
        s1 = double_part[i][0]
        o1 = double_part[i][1]
        s2 = double_part[i][5]
        o2 = double_part[i][6]          
        dpos = double_part[i][2:5]
   
        if s1==s2:
            # must be triplet
            # see case 2 of make_state_canonical in vs.py, namely
            # for same spin states, always order the orbitals
            S_d8_val[double_id] = 1
            data.append(np.sqrt(2.0));  row.append(double_id); col.append(double_id)
            if s1=='up':
                Sz_d8_val[double_id] = 1
            elif s1=='dn':
                Sz_d8_val[double_id] = -1
            count_triplet += 1

        elif s1=='dn' and s2=='up':
            print ('Error: d_double cannot have states with s1=dn, s2=up !')
            tstate = VS.get_state(VS.lookup_tbl[double_id])
            ts1 = tstate['hole1_spin']
            ts2 = tstate['hole2_spin']
            ts3 = tstate['hole3_spin']
            ts4 = tstate['hole4_spin'] 
            ts5 = tstate['hole5_spin']             
            torb1 = tstate['hole1_orb']
            torb2 = tstate['hole2_orb']
            torb3 = tstate['hole3_orb']
            torb4 = tstate['hole4_orb'] 
            torb5 = tstate['hole5_orb']             
            tx1, ty1, tz1 = tstate['hole1_coord']
            tx2, ty2, tz2 = tstate['hole2_coord']
            tx3, ty3, tz3 = tstate['hole3_coord']
            tx4, ty4, tz4 = tstate['hole4_coord']   
            tx5, ty5, tz5 = tstate['hole5_coord']               
            print ('Error state', double_id,ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,ts3,torb3,tx3,ty3,tz3,ts4,torb4,tx4,ty4,tz4,\
                   ts5,torb5,tx5,ty5,tz5)
            break

        elif s1=='up' and s2=='dn':
            if o1==o2: 
                if o1!='dxz' and o1!='dyz':
                    data.append(np.sqrt(2.0));  row.append(double_id); col.append(double_id)
                    S_d8_val[double_id]  = 0
                    Sz_d8_val[double_id] = 0
                    count_singlet += 1
                    
                # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                # instead of e1e1 and e2e2
                elif o1=='dxz':  # no need to consider e2='dyz' case
                    # generate paired e2e2 state:
                    if idx[i]==34:
                        slabel = [s1,'dyz']+dpos + [s2,'dyz']+dpos + hole34_part[i][0:5] + hole34_part[i][5:10]
                    elif idx[i]==24:
                        slabel = [s1,'dyz']+dpos + hole34_part[i][0:5] + [s2,'dyz']+dpos + hole34_part[i][5:10]
                    elif idx[i]==14:
                        slabel = hole34_part[i][0:5] + [s1,'dyz']+dpos + [s2,'dyz']+dpos + hole34_part[i][5:10]
                    elif idx[i]==13:
                        slabel = hole34_part[i][0:5] + [s1,'dyz']+dpos + hole34_part[i][5:10] + [s2,'dyz']+dpos 
                    elif idx[i]==23:
                        slabel = [s1,'dyz']+dpos + hole34_part[i][0:5] + hole34_part[i][5:10] + [s2,'dyz']+dpos 
                    elif idx[i]==12:
                        slabel = hole34_part[i][0:5] + hole34_part[i][5:10] + [s1,'dyz']+dpos + [s2,'dyz']+dpos                         

                    tmp_state = vs.create_state(slabel)
                    new_state,_,_ = vs.make_state_canonical(tmp_state)
                    e2 = VS.get_index(new_state)

                    data.append(1.0);  row.append(double_id);  col.append(double_id)
                    data.append(1.0);  row.append(e2); col.append(double_id)
                    AorB_d8_sym[double_id]  = 1
                    S_d8_val[double_id]  = 0                                                                            
                    Sz_d8_val[double_id] = 0
                    count_singlet += 1
                    data.append(1.0);  row.append(double_id);  col.append(e2)
                    data.append(-1.0); row.append(e2); col.append(e2)
                    AorB_d8_sym[e2] = -1
                    S_d8_val[e2]  = 0
                    Sz_d8_val[e2] = 0
                    count_singlet += 1


            else:
                if double_id not in count_list:
                    j, ph = find_singlet_triplet_partner_d_double(VS, double_part[i], idx[i], hole34_part[i])

                    # append matrix elements for singlet states
                    # convention: original state col i stores singlet and 
                    #             partner state col j stores triplet
                    data.append(1.0);  row.append(double_id); col.append(double_id)
                    data.append(-ph);  row.append(j); col.append(double_id)
                    S_d8_val[double_id]  = 0                                                                      
                    Sz_d8_val[double_id] = 0

                    #print "partner states:", i,j
                    #print "state i = ", s1, orb1, s2, orb2
                    #print "state j = ",'up',orb2,'dn',orb1

                    # append matrix elements for triplet states
                    data.append(1.0);  row.append(double_id); col.append(j)
                    data.append(ph);   row.append(j); col.append(j)
                    S_d8_val[j]  = 1
                    Sz_d8_val[j] = 0

                    count_list.append(j)

                    count_singlet += 1
                    count_triplet += 1
               

    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_d8_val, Sz_d8_val, AorB_d8_sym



def find_singlet_triplet_partner(VS, Ni_layer, Cu_layer, NiorCu, i, Ni_i, Cu_i,apz_layer, apz_i):
    '''
    For a given state (composed of Ni and Cu layer states) 
    find its partner state for each layer separately to form a singlet/triplet 
    Applies to general opposite-spin state, not nesessarily in d_double

    Parameters
    ----------
    VS: VS: VariationalSpace class from the module variational_space

    Returns
    -------
    phase: phase factor with which the partner state needs to be multiplied.

    Follow the rule of Ni followed by Cu, corresponding to VS   
    -------
    Ni_i: if Ni_i== [0,1],it means o1,o2 in Ni layer.   '''
    
    
    
    if NiorCu=='Ni':
#         print (Ni_i)
        mix_layer= apz_layer + Cu_layer
        if Ni_i== [0,1]:   
            slabel = [Ni_layer[5]]+ Ni_layer[1:5]+ [Ni_layer[0]]+ Ni_layer[6:10]+ mix_layer
        elif Ni_i==[0,2] :  
            slabel = [Ni_layer[5]]+ Ni_layer[1:5]+ mix_layer[0:5] + [Ni_layer[0]]+ Ni_layer[6:10]+ mix_layer[5:10]            
        elif Ni_i==[0,3] :  
            slabel = [Ni_layer[5]]+ Ni_layer[1:5]+ mix_layer + [Ni_layer[0]]+ Ni_layer[6:10] 
        elif Ni_i==[1,2] :  
            slabel = mix_layer[0:5]+ [Ni_layer[5]]+ Ni_layer[1:5]+  [Ni_layer[0]]+ Ni_layer[6:10]+ mix_layer[5:10] 
        elif Ni_i==[1,3] :  
            slabel = mix_layer[0:5]+ [Ni_layer[5]]+ Ni_layer[1:5]+ mix_layer[5:10]+ [Ni_layer[0]]+ Ni_layer[6:10]
        elif Ni_i==[2,3] :  
            slabel = mix_layer+ [Ni_layer[5]]+ Ni_layer[1:5]+ [Ni_layer[0]]+ Ni_layer[6:10]            
            
    elif NiorCu=='Cu':
#         print (Cu_i)
        mix_layer= Ni_layer + apz_layer
        if Cu_i== [0,1]:  
            slabel = [Cu_layer[5]]+ Cu_layer[1:5]+ [Cu_layer[0]]+ Cu_layer[6:10]+ mix_layer
        elif Cu_i==[0,2] :  
            slabel = [Cu_layer[5]]+ Cu_layer[1:5]+ mix_layer[0:5] + [Cu_layer[0]]+ Cu_layer[6:10]+ mix_layer[5:10]            
        elif Cu_i==[0,3] : 
            slabel = [Cu_layer[5]]+ Cu_layer[1:5]+ mix_layer + [Cu_layer[0]]+ Cu_layer[6:10] 
        elif Cu_i==[1,2] : 
            slabel = mix_layer[0:5]+ [Cu_layer[5]]+ Cu_layer[1:5]+  [Cu_layer[0]]+ Cu_layer[6:10]+ mix_layer[5:10] 
        elif Cu_i==[1,3] :  
            slabel =mix_layer[0:5]+ [Cu_layer[5]]+ Cu_layer[1:5]+ mix_layer[5:10]+ [Cu_layer[0]]+ Cu_layer[6:10]
        elif Cu_i==[2,3] :  
            slabel =mix_layer+ [Cu_layer[5]]+ Cu_layer[1:5]+ [Cu_layer[0]]+ Cu_layer[6:10]   
        
    #print(slabel)
    tmp_state = vs.create_state(slabel)
    partner_state, phase, _ = vs.make_state_canonical(tmp_state)
#     print(i,slabel,phase)
    return VS.get_index(partner_state), phase




def create_singlet_triplet_basis_change_matrix(VS, double_part, idx, hole34_part,d_Ni_double, d_Cu_double, NiorCu):
    '''
    Create a matrix representing the basis change to singlets/triplets. The
    columns of the output matrix are the new basis vectors. 
    The Hamiltonian transforms as U_dagger*H*U. 

    Parameters
    ----------
    phase: dictionary containing the phase factors created with
        hamiltonian.create_phase_dict.
    VS: VariationalSpace class from the module variational_space. Should contain
        only zero-magnon states.

    Returns
    -------
    U: matrix representing the basis change to singlets/triplets in
        sps.coo format.
    '''
    data = []
    row = []
    col = []
    
    #count_upup, count_updn, count_dnup, count_dndn = count_VS(VS)
    #print count_upup, count_updn, count_dnup, count_dndn
    
    count_singlet = 0
    count_triplet = 0
    
    # store index of partner state to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []
    
    # denote if the new state is singlet (0) or triplet (1)
    S_val    = np.zeros(VS.dim, dtype=int)
    Sz_val   = np.zeros(VS.dim, dtype=int)
    AorB_sym = np.zeros(VS.dim, dtype=int)
    
    for i in range(0,VS.dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        s4 = start_state['hole4_spin']        
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        orb4 = start_state['hole4_orb']        
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']          
        x3, y3, z3 = start_state['hole3_coord']
        x4, y4, z4 = start_state['hole4_coord']          
        
        
        if NiorCu=='Ni':
            d_double = d_Ni_double
                 

 
        # N_u means holes stay in Ni layer and N_d means holes stay in Cu layer
    
        if NiorCu=='Cu':
            d_double = d_Cu_double    

            
        # get states in Ni and Cu layers separately and how many orbs
        Ni_layer, N_Ni, Cu_layer, N_Cu, Ni_i, Cu_i,apz_layer, N_H,apz_i = util.get_NiCu_layer_orbs(start_state)

        
        # calculate singlet or triplet only if the layer exist two holes        
        if (not ((N_Ni== 2 and NiorCu=='Ni') or (N_Cu== 2 and NiorCu=='Cu'))) and (i not in d_double):
#         if i not in d_double:            
            data.append(np.sqrt(2.0)); row.append(i); col.append(i)         
        
        elif i not in count_list:
            if i in d_double:
                i2 = d_double.index(i)
                j, ph = find_singlet_triplet_partner_d_double(VS, double_part[i2], idx[i2], hole34_part[i2])
#                 print(i,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,ph) 
                
                s1 = double_part[i2][0]
                o1 = double_part[i2][1]
                s2 = double_part[i2][5]
                o2 = double_part[i2][6]          
                dpos = double_part[i2][2:5]
            else:
                j, ph = find_singlet_triplet_partner(VS, Ni_layer, Cu_layer, NiorCu,i, Ni_i, Cu_i,H_layer, H_i)
#                 print(i,s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4,ph) 
                
                if NiorCu=='Ni':
                    s1 = Ni_layer[0]
                    s2 = Ni_layer[5]
                    o1 = Ni_layer[1]
                    o2 = Ni_layer[6] 
                elif NiorCu=='Cu':
                    s1 = Cu_layer[0]
                    s2 = Cu_layer[5]
                    o1 = Cu_layer[1]
                    o2 = Cu_layer[6]                 
            
            #print "partner states:", i,j
            #print "state i = ", s1, orb1, s2, orb2
            #print "state j = ",'up',orb2,'dn',orb1
            count_list.append(j)
  
            if j==i:
                                
                if s1==s2:
                    # must be triplet
                    data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                    S_val[i] = 1
                    if s1=='up':
                        Sz_val[i] = 1
                    elif s1=='dn':
                        Sz_val[i] = -1
                    count_triplet += 1

                else:

                    # only possible other states for j=i 
                    assert(s1=='up' and s2=='dn' and o1==o2)
                    
                    # get state as (e1e1 +- e2e2)/sqrt(2) for A and B sym separately 
                    # instead of e1e1 and e2e2
                    if o1!='dxz' and o1!='dyz':
                        data.append(np.sqrt(2.0));  row.append(i); col.append(i)
                        S_val[i]  = 0
                        Sz_val[i] = 0
                        count_singlet += 1
                        
                    elif o1==o2=='dxz':  # no need to consider e2='dyz' case
                    # generate paired e2e2 state:
                        if idx[i2]==34:
                            slabel = [s1,'dyz']+dpos + [s2,'dyz']+dpos + hole34_part[i2][0:5] + hole34_part[i2][5:10]
                        elif idx[i2]==24:
                            slabel = [s1,'dyz']+dpos + hole34_part[i2][0:5] + [s2,'dyz']+dpos + hole34_part[i2][5:10]
                        elif idx[i2]==14:
                            slabel = hole34_part[i2][0:5] + [s1,'dyz']+dpos + [s2,'dyz']+dpos + hole34_part[i2][5:10]
                        elif idx[i2]==13:
                            slabel = hole34_part[i2][0:5] + [s1,'dyz']+dpos + hole34_part[i2][5:10] + [s2,'dyz']+dpos 
                        elif idx[i2]==23:
                            slabel = [s1,'dyz']+dpos + hole34_part[i2][0:5] + hole34_part[i2][5:10] + [s2,'dyz']+dpos 
                        elif idx[i2]==12:
                            slabel = hole34_part[i2][0:5] + hole34_part[i2][5:10] + [s1,'dyz']+dpos + [s2,'dyz']+dpos                         

                        tmp_state = vs.create_state(slabel)
                        new_state,_,_ = vs.make_state_canonical(tmp_state)
                        e2 = VS.get_index(new_state)

                        data.append(1.0);  row.append(i);  col.append(i)
                        data.append(1.0);  row.append(e2); col.append(i)
                        AorB_sym[i]  = 1
                        S_val[i]  = 0                                                                            
                        Sz_val[i] = 0
                        count_singlet += 1
                        data.append(1.0);  row.append(i);  col.append(e2)
                        data.append(-1.0); row.append(e2); col.append(e2)
                        AorB_sym[e2] = -1
                        S_val[e2]  = 0
                        Sz_val[e2] = 0
                        count_singlet += 1


            else:
                
                # append matrix elements for singlet states
                # convention: original state col i stores singlet and 
                #             partner state col j stores triplet
                data.append(1.0);  row.append(i); col.append(i)
                data.append(-ph);  row.append(j); col.append(i)
                S_val[i]  = 0                                                                      
                Sz_val[i] = 0

                #print "partner states:", i,j
                #print "state i = ", s1, orb1, s2, orb2
                #print "state j = ",'up',orb2,'dn',orb1

                # append matrix elements for triplet states
                data.append(1.0);  row.append(i); col.append(j)
                data.append(ph);   row.append(j); col.append(j)
                S_val[j]  = 1
                Sz_val[j] = 0
#                 print (i,j)

                count_singlet += 1
                count_triplet += 1 

     
  

    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), S_val, Sz_val, AorB_sym


def create_bonding_anti_bonding_basis_change_matrix(VS):
    
    dim = VS.dim
    data = []
    row = []
    col = []   
    
    bonding_val = np.zeros(VS.dim, dtype=int)    
    
    # store index of partner state to avoid double counting
    # otherwise, when arriving at i's partner j, its partner would be i
    count_list = []    
    
    for i in range(0,VS.dim):
        start_state = VS.get_state(VS.lookup_tbl[i])
        s1 = start_state['hole1_spin']
        s2 = start_state['hole2_spin']
        s3 = start_state['hole3_spin']
        s4 = start_state['hole4_spin']            
        orb1 = start_state['hole1_orb']
        orb2 = start_state['hole2_orb']
        orb3 = start_state['hole3_orb']
        orb4 = start_state['hole4_orb']       
        x1, y1, z1 = start_state['hole1_coord']
        x2, y2, z2 = start_state['hole2_coord']          
        x3, y3, z3 = start_state['hole3_coord']
        x4, y4, z4 = start_state['hole4_coord']          
        
        slabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4]

        
        #Two layers of Cu and Ni exchange position and when z=1 in apz orb,it's still itself
        
        slabel2 = [s1,orb1,x1,y1,2-z1,s2,orb2,x2,y2,2-z2,s3,orb3,x3,y3,2-z3,s4,orb4,x4,y4,2-z4]
        tmp_state = vs.create_state(slabel2)
        partner_state,phase,_ = vs.make_state_canonical(tmp_state)      
#         print (phase)
        j = VS.get_index(partner_state)        
        
        if j==i:
            data.append(np.sqrt(2.0)); row.append(i); col.append(i)
            bonding_val[i] = 0 
        else:
#             start_state2 = VS.get_state(VS.lookup_tbl[i])
#             s1 = start_state2['hole1_spin']
#             s2 = start_state2['hole2_spin']
#             s3 = start_state2['hole3_spin']
#             s4 = start_state2['hole4_spin']            
#             orb1 = start_state2['hole1_orb']
#             orb2 = start_state2['hole2_orb']
#             orb3 = start_state2['hole3_orb']
#             orb4 = start_state2['hole4_orb']       
#             x1, y1, z1 = start_state2['hole1_coord']
#             x2, y2, z2 = start_state2['hole2_coord']          
#             x3, y3, z3 = start_state2['hole3_coord']
#             x4, y4, z4 = start_state2['hole4_coord'] 
            
#             start_state3 = VS.get_state(VS.lookup_tbl[j]) 
#             if orb1=='d3z2r2' and orb2=='dx2y2' and orb3=='d3z2r2' and orb4=='dx2y2':
#                 print (start_state2)
#                 print (start_state3)            
            if i not in count_list:
                # append matrix elements for bonding
                # convention: original state col i stores bonding and 
                #             partner state col j stores anti-bonding
                data.append(1.0);  row.append(i); col.append(i)
                data.append(phase);  row.append(j); col.append(i)
                bonding_val[i] = 1          


                # append matrix elements for anti-bonding
                data.append(1.0);  row.append(i); col.append(j)
                data.append(-phase);   row.append(j); col.append(j)
                bonding_val[j] = -1


                count_list.append(j)

    return sps.coo_matrix((data,(row,col)),shape=(VS.dim,VS.dim))/np.sqrt(2.0), bonding_val       

# def print_VS_after_basis_change(VS,S_val,Sz_val):
#     print ('print_VS_after_basis_change:')
#     for i in range(0,VS.dim):
#         state = VS.get_state(VS.lookup_tbl[i])
#         ts1 = state['hole1_spin']
#         ts2 = state['hole2_spin']
#         torb1 = state['hole1_orb']
#         torb2 = state['hole2_orb']
#         tx1, ty1, tz1 = state['hole1_coord']
#         tx2, ty2, tz2 = state['hole2_coord']
#         #if ts1=='up' and ts2=='up':
#         if torb1=='dx2y2' and torb2=='px':
#             print (i, ts1,torb1,tx1,ty1,tz1,ts2,torb2,tx2,ty2,tz2,'S=',S_val[i],'Sz=',Sz_val[i])
            
