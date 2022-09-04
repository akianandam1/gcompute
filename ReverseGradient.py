import torch
from RevisedNumericalSolver import torchstate
#from equalmass import values

# Perturbs input vector using normal distribution
# takes in float standard deviation
# Requires floats
def perturb(vec, std):
    return torch.tensor([torch.normal(mean = vec[0], std = torch.tensor(std)),
                         torch.normal(mean = vec[1], std = torch.tensor(std)),
                         0.0,
                         torch.normal(mean = vec[3], std = torch.tensor(std)),
                         torch.normal(mean = vec[4], std = torch.tensor(std)),
                         0.0,
                         torch.normal(mean = vec[6], std = torch.tensor(std)),
                         torch.normal(mean = vec[7], std = torch.tensor(std)),
                         0.0,
                         torch.normal(mean = vec[9], std = torch.tensor(std)),
                         torch.normal(mean = vec[10], std = torch.tensor(std)),
                         0.0,
                         torch.normal(mean = vec[12], std = torch.tensor(std)),
                         torch.normal(mean = vec[13], std = torch.tensor(std)),
                         0.0,
                         torch.normal(mean = vec[15], std = torch.tensor(std)),
                         torch.normal(mean = vec[16], std = torch.tensor(std)),
                         0.0, ], requires_grad = True)

def perturb2(vec, std):
    return torch.tensor([torch.normal(mean = vec[0], std = torch.tensor(std))
                         #,torch.normal(mean = vec[1], std = torch.tensor(std))
                         #,torch.normal(mean = vec[2], std = torch.tensor(std))
                         ], requires_grad = True)
# Epoch ~6400 at .001 time length 15

# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
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




# Input vec is in form x_1, v_1, v_2; m_1=m_2=m_3=1
def reverse_grad(a, b, c, lr, time_step, num_epochs, max_period):
    x_1 = torch.tensor([a], requires_grad = True)
    x_1 = perturb2(x_1, .001)
    v_1 = torch.tensor([b], requires_grad = True)
    v_1 = perturb2(v_1, .001)
    v_2 = torch.tensor([c], requires_grad = True)
    v_2 = perturb2(v_2, .001)


    print("Begun")
    i = 0
    while i < num_epochs:
        input_vec = torch.stack((
            x_1,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            v_1,
            torch.tensor([0]),
            torch.tensor([0]),
            v_2,
            torch.tensor([0]),
            torch.tensor([0]),
            -v_1-v_2,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([1]),

        )).flatten()
        #print(input_vec)
        #input_vec = torch.tensor([vec[0], 0, 0, 1, 0, 0, 0, 0, 0, 0, vec[1], 0, 0, vec[2], 0, 0, -vec[1]-vec[2], 0, 1, 1, 1], requires_grad = True)
        #vec.retain_grad()
        data_set = torchstate(input_vec, time_step, max_period, "rk4")
        #vec.retain_grad()
        first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), time_step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), time_step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), time_step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0], second_particle_state) + nearest_position(3, data_set[0], third_particle_state)

        #vec.retain_grad()
        loss.backward()
        print(x_1.grad, v_1.grad, v_2.grad)
        with torch.no_grad():
            x_1 += x_1.grad * lr
            v_1 += v_1.grad * lr
            v_2 += v_2.grad * lr
        x_1.grad.zero_()
        v_1.grad.zero_()
        v_2.grad.zero_()

        with open("ReverseGradientPoints.txt", "a") as file:
            file.write(f"{x_1.item()},{v_1.item()},{v_2.item()}\n")
        print(f"{x_1.item()},{v_1.item()},{v_2.item()}\n")
        print(input_vec)
        print(f"Epoch: {i}")
        i += 1



# Takes in v_1 and v_2
def reverse_grad2(a, b, lr, time_step, num_epochs, max_period):
    v_1 = torch.tensor([a], requires_grad = True)
    v_1 = perturb2(v_1, .01)
    v_2 = torch.tensor([b], requires_grad = True)
    v_2 = perturb2(v_2, .01)


    print("Begun")
    i = 0
    while i < num_epochs:
        input_vec = torch.stack((
            torch.tensor([-1]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            v_1,
            v_2,
            torch.tensor([0]),
            v_1,
            v_2,
            torch.tensor([0]),
            -2*v_1,
            -2*v_2,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([1]),

        )).flatten()
        #print(input_vec)
        #input_vec = torch.tensor([vec[0], 0, 0, 1, 0, 0, 0, 0, 0, 0, vec[1], 0, 0, vec[2], 0, 0, -vec[1]-vec[2], 0, 1, 1, 1], requires_grad = True)
        #vec.retain_grad()
        data_set = torchstate(input_vec, time_step, max_period, "rk4")
        #vec.retain_grad()
        first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), time_step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), time_step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), time_step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0], second_particle_state) + nearest_position(3, data_set[0], third_particle_state)

        #vec.retain_grad()
        loss.backward()
        print(v_1.grad, v_2.grad)
        with torch.no_grad():
         
            v_1 += v_1.grad * lr
            v_2 += v_2.grad * lr
     
        v_1.grad.zero_()
        v_2.grad.zero_()

        with open("ReverseGradientPoints.txt", "a") as file:
            file.write(f"{v_1.item()},{v_2.item()}\n")
        print(f"{v_1.item()},{v_2.item()}\n")
        print(input_vec)
        print(f"Epoch: {i}")
        i += 1


#1.0000 1.0000 1.0000 -1.325626981682458e+00 -8.933877752879044e-01 -2.885702941263346e-01 9.199307755830397e+00 3.831608876556280e-01
# -0.37200864090742 	1.21800411067968 	0.45310805383360 	7.53971451331775

#v = torch.tensor([-1.325626981682458e+00, -8.933877752879044e-01, -2.885702941263346e-01], requires_grad = True)
#v = perturb2(v, .01)


if __name__ == "__main__":
    reverse_grad(-1.3176461458206177,-0.8931861519813538,-0.31467458605766296, .001, .01, 500, int(9.199307755830397e+00)+2)
#reverse_grad(-1.325626981682458e+00, -8.933877752879044e-01, -2.885702941263346e-01, .001, .01, 500, int(9.199307755830397e+00)+2)
