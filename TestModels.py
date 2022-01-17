import torch
from BallDrop import BallDropNet
import numpy as np
import matplotlib.pyplot as plt


def drop(time, initial_velocity, initial_height):
    return -4.9 * time ** 2 + initial_velocity * time + initial_height


# Returns total time the ball will spend in the air given initial conditions
def time_taken(initial_velocity, initial_height):
    return (-initial_velocity - np.sqrt(initial_velocity ** 2 + 4 * 4.9 * initial_height)) / (2 * -4.9)


# Imports BallDrop model
imported_model = BallDropNet(3,1000,1)
imported_model.load_state_dict(torch.load('Models\ThirdModel.pth'))


# Function to plot real curves vs network's predicted curves
def PlotCurves(model, axis,v,h):
    t = np.linspace(0, time_taken(v, h), 1000)
    y_values = []
    predictions = []
    for time in t:
        y_values.append(drop(time, v, h))
        predictions.append(model(torch.tensor([time, v, h], dtype = torch.float32)).item())
    y_values = np.array(y_values, dtype = 'float32')
    predictions = np.array(predictions, dtype = 'float32')
    axis.plot(t, y_values)
    axis.plot(t, predictions)


# 4 plots
figure, axis = plt.subplots(2, 2)

# Plots 4 random curves
PlotCurves(imported_model, axis[0,0], -20, 5)
PlotCurves(imported_model, axis[1,0], 8, 51)
PlotCurves(imported_model, axis[0,1], 70, 5)
PlotCurves(imported_model, axis[1,1], -3, 87)

plt.show()