# Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import random
#from . import sensor
from sensor import Sensor

FRAMES = 1000
MS_PER_PLOT = 20
COLORS = cm.get_cmap('tab20b', FRAMES)
STD_DEVS = 3

import matplotlib.patches as patches

def plot_covariance_ellipse(position, covariance, ax, n_std=STD_DEVS, **kwargs):
    """
    Add a covariance ellipse to the given axis.

    - position: center of the ellipse (x, y)
    - covariance: 2x2 covariance matrix
    - ax: matplotlib Axes
    - n_std: number of standard deviations (2 = ~95% confidence)
    - kwargs: extra arguments for ellipse (e.g., alpha, edgecolor)
    """
    from matplotlib.transforms import Affine2D

    if covariance.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2")

    eigenvals, eigenvecs = np.linalg.eigh(covariance)

    # Sort eigenvalues (and corresponding vectors) largest first
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    # Compute angle from largest eigenvector
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))

    # Width and height of ellipse = 2 * sqrt(eigenvalue) * n_std
    width, height = 2 * n_std * np.sqrt(eigenvals)

    ellipse = patches.Ellipse(xy=position, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def main():
    targets = set()                                 # set of targets
    sensors = set()                                 # set of sensors

    # Object specific variables
    v: float = 300                                  # m/s
    q: float = 9                                    # m/s^2
    A: float = v**(2)/q                             # m
    omega: float = q/(2*v)                          # 1/s

    # General variables
    T: float = 2 * np.pi / omega                    # one period
    t: np.ndarray = np.linspace(0, T, FRAMES)       # shape: (1000,)

    # More object specific variables
    # Movement
    r = np.ndarray((np.shape(t)[0], 2))
    x = r[:,0] = A * np.sin(omega*t)                # x(t)
    y = r[:,1] = A * np.sin(2*omega*t)              # y(t)

    # Velocity
    r_dot: np.ndarray = np.ndarray(np.shape(r))
    x_dot = r_dot[:,0] = v*np.cos(omega*t)/2
    y_dot = r_dot[:,1] = v*np.cos(2*omega*t)
    velocity: np.ndarray = np.linalg.norm(r_dot, axis=1)                    # TODO: maybe plot

    # Acceleration
    r_dot_dot: np.ndarray = np.ndarray(np.shape(r))
    x_dot_dot = r_dot_dot[:,0] = -q*np.sin(omega*t)/4
    y_dot_dot = r_dot_dot[:,1] = -q*np.sin(2*omega*t)
    acceleration: np.ndarray = np.linalg.norm(r_dot_dot, axis=1)

    x_k: np.ndarray = np.array([x,y,x_dot,y_dot,x_dot_dot,y_dot_dot]).T        # shape: (6,1)

    # Main
    # Sensor Init
    s1 = Sensor(color='red')
    sensors.add(s1)
    r_s = s1.get_position()
    print(r_s)
    # Object Init
    # TODO

    # kalman = sth sth arraylike (len(sensors), len(objects))

    # Plot
    fig, ax = plt.subplots()
    buffer_x = (np.max(x) - np.min(x)) / 10
    buffer_y = (np.max(y) - np.min(y)) / 10
    min_x = np.min(x) - buffer_x
    max_x = np.max(x) + buffer_x
    min_y = np.min(y) - buffer_y
    max_y = np.max(y) + buffer_y

    s1_pos = ax.plot(r_s[0], r_s[1], 'ro', label = 'sensor1')                       
    # o1_pos = ax.plot(0, 0, color='orange', label='object1')

    # Trajectory
    o1_line = ax.plot(x[0], y[0], label='object1 trace')[0]           # object 1
    cartesian_measurements = []
    cartesian_measurements_scatter = ax.scatter([], [], marker='x', color='green', s=5, label='cartesian_measurements')
    polar_measurements = []
    polar_measurements_scatter = ax.scatter([], [], marker='x', color='red', s=5, label='polar_measurements')
    predictions = []
    predictions_scatter = ax.scatter([], [], marker='x', color='pink', s=5, label='prediction')
    ax.set(xlim=[min_x,max_x], ylim=[min_y,max_y], xlabel='x', ylabel='y')
    ax.legend()

    def init():
        o1_line.set_data([], [])
        cartesian_measurements_scatter.set_offsets(np.empty((0, 2)))
        polar_measurements_scatter.set_offsets(np.empty((0, 2)))
        predictions_scatter.set_offsets(np.empty((0, 2)))
        # If s1_pos is an artist that needs to be reset, do it here as well, e.g.:
        # s1_pos.set_offsets([])
        
        # Return all artists that will be updated
        return o1_line, s1_pos, cartesian_measurements_scatter, polar_measurements_scatter, predictions_scatter

    # Update function - does all the calculations
    def update(frame):
        # Object movement
        o1_x_plot = x[:frame]                      # all values until 
        o1_y_plot = y[:frame]                      # current frame
        o1_line.set_xdata(o1_x_plot)
        o1_line.set_ydata(o1_y_plot)

        sensor_data = set()
        for sensor in sensors:
            # Prediction
            if frame > 0:
                # print(cartesian_measurements)
                prediction = sensor.predict()
                predictions.append(prediction[0][:2])
                covariance = prediction[1]
                positional_cov = covariance[:2,:2]
                # print("covariance:", np.shape(covariance), "\n", positional_cov)
                pred_to_array = np.array(predictions)
                predictions_scatter.set_offsets(pred_to_array)
                sensor_data.add(predictions_scatter)

                # Plot covariance ellipse for current prediction
                # Clear old ellipses if necessary
                for e in ax.patches[:]:
                    e.remove()

                plot_covariance_ellipse(
                    position=prediction[0][:2],
                    covariance=positional_cov,
                    ax=ax,
                    n_std=STD_DEVS,
                    edgecolor='red',
                    facecolor='none',
                    linewidth=1.5)

            if frame % sensor.time_between_measurements == 0:
                # Measurement
                current_state = x_k[frame-3,:]        # no idea why, but subtracting this value works bests
                # Cartesian measurement
                cartesian_measurement = sensor.measure_cartesian(current_state)
                cartesian_measurements.append(cartesian_measurement)
                cartesian_measurements_scatter.set_offsets(np.array(cartesian_measurements))
                sensor_data.add(cartesian_measurements_scatter)
                # Polar measurement
                polar_return = sensor.measure_polar(current_state)
                polar_measurement = polar_return[0][0]*np.array((np.cos(polar_return[0][1]),np.sin(polar_return[0][1]))) + polar_return[1]
                polar_measurements.append(polar_measurement)
                polar_measurements_scatter.set_offsets(np.array(polar_measurements))
                sensor_data.add(polar_measurements_scatter)
                
                # Filtering
                # new_measurement = np.array((cartesian_measurements[int(frame/sensor.time_between_measurements)]))
                new_measurement = np.array((polar_measurements[int(frame/sensor.time_between_measurements)]))
                sensor.filter(new_measurement)
        return o1_line, s1_pos, sensor_data

    anim = animation.FuncAnimation(fig=fig, func=update, init_func=init, frames=FRAMES, interval=MS_PER_PLOT)
    plt.show()

if __name__ == '__main__': main()