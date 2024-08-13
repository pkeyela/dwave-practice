#!/usr/bin/env python
# coding: utf-8

# In[103]:


import random
import math
import numpy as np

from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler


def generate_open_ran_parameters(B, K_max, U):
    # Generate gNodeBs
    gNodeBs = {i+1: {"RBs": [], "users": [], "gXY": (0, 0)} for i in range(B)}

    # Generate random coordinates for gNodeBs within a defined area (e.g., 1000x1000)
    for i in range(1, B+1):
        gNodeBs[i]["gXY"] = (random.uniform(200, 1000), random.uniform(200, 1000))
    
    # Distribute resource blocks among gNodeBs non-uniformly
    remaining_RBs = K_max
    for i in range(1, B):
        K_i = random.randint(1, remaining_RBs - (B - i))
        gNodeBs[i]["RBs"] = list(range(K_max - remaining_RBs + 1, K_max - remaining_RBs + K_i + 1))
        gNodeBs[i]["Kb"] = K_i
        remaining_RBs -= K_i
    
    gNodeBs[B]["RBs"] = list(range(K_max - remaining_RBs + 1, K_max + 1))
    gNodeBs[B]["Kb"] = remaining_RBs

    # Distribute users among gNodeBs non-uniformly
    remaining_users = U
    index = 0
    for i in range(1, B):
        U_i = random.randint(1, remaining_users - (B - i))
        gNodeBs[i]["Ub"] = U_i
        for j in range(index, index + U_i):
            slice_type = 0 if random.random() < 0.5 else 1
            distance = random.uniform(0, 250)
            user_coords = (
                gNodeBs[i]["gXY"][0] + random.uniform(-distance, distance),
                gNodeBs[i]["gXY"][1] + random.uniform(-distance, distance)
            )
            gNodeBs[i]["users"].append((j, slice_type, user_coords))
        index += U_i
        remaining_users -= U_i
    
    gNodeBs[B]["Ub"] = remaining_users
    
    for j in range(index, index + remaining_users):
        slice_type = 0 if random.random() < 0.5 else 1
        distance = random.uniform(0, 300)  # 
        user_coords = (
            gNodeBs[B]["gXY"][0] + random.uniform(-distance, distance),
            gNodeBs[B]["gXY"][1] + random.uniform(-distance, distance)
        )
        gNodeBs[B]["users"].append((j, slice_type, user_coords))
    
    gNodeBs["B"] = B
    gNodeBs["U"] = U
    gNodeBs["Kmax"] = K_max

    return gNodeBs


# In[106]:


# Print the generated data
def print_network(gNodeBs):
    print("Generated gNodeBs with RBs and Users:")
    for gNodeB, data in gNodeBs.items():
        print(f"gNodeB {gNodeB}:")
        print(f"  No RBs: {data['Kb']}")
        print(f"  RBs: {data['RBs']}")
        print(f"  No Users: {data['Ub']}")
        print(f"  Users: {data['users']}")


# In[107]:


def dbm_to_watt(dbm):
    """
    Convert power from dBm to Watts.

    Parameters:
    dbm (float): Power in dBm.

    Returns:
    float: Power in Watts.
    """
    return 10 ** (dbm / 10) * 1e-3


def calculate_channel_gain(frequency_hz=3.5e6, distance_m=1):
    """
    Calculate the channel gain using the Free-space Path Loss (FSPL) model.

    Parameters:
    frequency_hz (float): Frequency of the signal in Hz.
    distance_m (float): Distance between the gNodeB and the user in meters.

    Returns:
    float: Channel gain G_{u_b k}.
    """
    c = 3 * 10**8  # Speed of light in meters/second
    wavelength = c / frequency_hz
    path_loss_linear = (wavelength / (4 * np.pi * distance_m)) ** 2
    return path_loss_linear

def calculate_distance(gNodeB_coords, user_coords):
    """
    Calculate the Euclidean distance between a gNodeB and a user.
    
    Parameters:
    - gNodeB_coords: tuple, the (x, y) coordinates of the gNodeB
    - user_coords: tuple, the (x, y) coordinates of the user
    
    Returns:
    - distance: float, the Euclidean distance between the gNodeB and the user
    """
    x1, y1 = gNodeB_coords
    x2, y2 = user_coords
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# In[108]:


# Create CQM object
def build_cqm(gNodeBs, K_max, R_min, D_max, W, P_sub, sigma_sq, lambda_sub = 100, packet_length_embb = 400, packet_length_urllc = 120):
    '''Builds the CQM for our problem'''
    
    def data_rate(user_dist):
        Gubk = calculate_channel_gain(distance_m=user_dist)
        return W * np.log2(1 + (P_sub * Gubk) / sigma_sq)

    # Initialize the CQM object
    cqm = ConstrainedQuadraticModel()
    
    # Constraint 1
    for gnb in range(1, gNodeBs['B']+1):
        for user in gNodeBs[gnb]['users']:
            cqm.add_constraint(sum(Binary(f"x_k{k}b{gnb}u{user[0]}") 
                                   for k in range(1,gNodeBs['Kmax']+1)) 
                               <= gNodeBs['Kmax'],
                               label=f"C1_b{gnb}u{user[0]}")
    # Constraint 2
    for k in range(1, gNodeBs['Kmax']+1):
        for gnb in range(1, gNodeBs['B']+1):        
            cqm.add_constraint(sum(Binary(f"x_k{k}b{gnb}u{user[0]}")
                                   for user in gNodeBs[gnb]['users'])
                               <= 1,
                               label=f"C2k{k}b{gnb}")
    
    # Constraint 3
    for gnb in range(1, gNodeBs['B']+1):
        for user in gNodeBs[gnb]['users']:
            rhs = quicksum(Binary(f"x_k{k}b{gnb}u{user[0]}")
                                   for k in gNodeBs[gnb]['RBs'])
            
            Kprime = set(range(1, gNodeBs['Kmax']+1)) - set(gNodeBs[gnb]['RBs'])
            
            for kpr in Kprime:
                cqm.add_constraint(Binary(f"x_k{kpr}b{gnb}u{user[0]}") * gNodeBs[gnb]['Kb'] - rhs <= 0,
                               label=f"C3k{kpr}b{gnb}u{user}")
    
    # Constraint 4
    obj = 0
    for gnb in range(1, gNodeBs['B']+1):
        for user in gNodeBs[gnb]['users']:
            # print(f"User {user[0]} coordinates: {user[2]}")
            rub = 0
            for k in gNodeBs[gnb]['RBs']:
                dist_gnb_user = calculate_distance(gNodeB_coords =gNodeBs[gnb]['gXY'],
                                                   user_coords = user[2])
                r_kub = data_rate(dist_gnb_user)
                
                rub += Binary(f"x_k{kpr}b{gnb}u{user[0]}") * r_kub
            
            gNBprime = set(range(1, gNodeBs['B']+1)) - set([gnb])
            
            for gnbpr in gNBprime:
                for kpr in gNodeBs[gnbpr]['RBs']:
                    dist_gnbpr_user = calculate_distance(gNodeBs[gnbpr]['gXY'], user[2])
                    r_kprub = data_rate(dist_gnbpr_user)

                    rub += Binary(f"x_k{kpr}b{gnb}u{user[0]}") * r_kprub
                    
            cqm.add_constraint(rub >= R_min, label = f"C4_b{gnb}_u{user[0]}")
            
            if user[1]:
                lambda_s = lambda_sub * packet_length_embb
            else:
                lambda_s = lambda_sub * packet_length_embb
            
            # Constraint 5
            cqm.add_constraint(D_max* (rub - lambda_s) <= 1., label = f"C5_b{gnb}_u{user[0]}")

            obj += rub

    cqm.set_objective(-obj)

    return cqm


# In[111]:


def sample_cqm(cqm):

    # Define the sampler as LeapHybridCQMSampler
    sampler = LeapHybridCQMSampler()

    # Sample the ConstrainedQuadraticModel cqm and store the result in sampleset
    sampleset = sampler.sample_cqm(cqm=cqm)

    return sampleset


if __name__ == '__main__':
    #### PARAMETERS ####
    # 5G Frequency 
    frequency_hz = 3.5 * 10**9 #[Hz]

    # Input parameters
    B = 5  # Number of gNodeBs
    K_max = 100  # Total number of RBs for all gNodeBs
    U = 50  # Total number of users

    # Parameters for simulation
    W = 180 * 10**3  # Bandwidth of an RB in Hz
    P_sub = 30  # Transmit power of gNodeB in dBm
    sigma_squared = -114  # Power of AWGN in dBm

    lambda_sub = 100  # Packet arriving rate per end-user in packets/s
    packet_length_embb = 400  # Packet length for eMBB end-user in bits
    packet_length_urllc = 120  # Packet length for URLLC end-user in bits

    R_min = 100 * 10**3  # Minimum data rate for eMBB end-user in bps
    D_max = 10 * 10**-3  # Maximum delay for URLLC end-user in seconds

    # Generate the parameters
    gNodeBs = generate_open_ran_parameters(B, K_max, U)

    # CONVERSION OF PARAMETERS
    P_watt = dbm_to_watt(P_sub)
    noise_watt = dbm_to_watt(sigma_squared)

    # Build CQM
    cqm =  build_cqm(gNodeBs, K_max, R_min, D_max, W, P_watt, noise_watt)

    # Run CQM on hybrid solver
    sampleset = sample_cqm(cqm)

    print(sampleset)

