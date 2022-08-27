from RevisedNumericalSolver import torchstate
import torch
import time

def nearest_position(particle, state1, state2):
    mse = torch.nn.L1Loss()
    if particle == 1:
        return mse(state1[:3], state2[:3]) + mse(state1[9:12], state2[9:12])
    elif particle == 2:
        return mse(state1[3:6], state2[3:6]) + mse(state1[12:15], state2[12:15])
    elif particle == 3:
        return mse(state1[6:9], state2[6:9]) + mse(state1[15:18], state2[15:18])
    else:
        print("bad input")


# Finds the most similar state to the initial position in a data set
def nearest_position_state(particle, state, data_set, min, max, time_step):
    i = min
    max_val = torch.tensor([100000000])
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1

    print(f"Time: {index*time_step}")
    return index



def optimize(vec, m_1, m_2, m_3, time_step, max_period, num_periods, num_epochs, method, optimizer):
    
    
    optimizer.zero_grad()
    i = 0
    period_index = int(5.6664/time_step)
    while i < num_epochs:
              
        k = num_periods-1
        input_vec = torch.cat((vec, torch.tensor([m_1,m_2,m_3])))
        data_set = torchstate(input_vec, time_step, max_period, method)
        print(f"Max Period: {max_period}")
        first_index = nearest_position_state(1, data_set[0], data_set, k*period_index+300, (k+1)*(period_index+100), time_step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, k*period_index+300, (k+1)*(period_index+100), time_step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, k*period_index+300, (k+1)*(period_index+100), time_step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0], second_particle_state) + nearest_position(3, data_set[0], third_particle_state)

        with open("optimizelog2.txt", "a") as file:
            file.write(f"{input_vec},{i},{loss}\\n")

        print(loss)

        loss.backward()

        # Updates input vector
        optimizer.step()
        print(input_vec)
        print(f"Epoch: {i}")
        optimizer.zero_grad()  
            
            
               
        i += num_periods
        
        
m_1 = 1
m_2 = 1
m_3 = 0.75
vec = torch.tensor([-0.9819,  0.0308,  0.0000,  0.9667,  0.0287,  0.0000, -0.0132,  0.0095,
         0.0000,  0.4219,  0.2533,  0.0000,  0.4213,  0.2707,  0.0000, -1.1225,
        -0.6964,  0.0000], requires_grad = True)

optimizer = torch.optim.SGD([vec], lr=.0000001)
optimize(vec, m_1, m_2, m_3, .0008, 25, 4, 520, "dopri5", optimizer)
    
