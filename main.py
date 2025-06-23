# Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import random
#from . import sensor
from sensor import Sensor

FRAMES = 1000
MS_PER_PLOT = 20
COLORS = cm.get_cmap('tab20b', FRAMES)

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

    x_k: np.ndarray = np.array([x,y,x_dot,y_dot,x_dot_dot,y_dot_dot]).T
    print(x_k)
    print(np.shape(x_k))

    # Main
    # Sensor Init
    print(np.shape(np.eye(2))==np.shape(np.zeros_like(np.eye(2))))
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
    o1_pos = ax.plot(0, 0, color='orange', label='object1')

    # Trajectory
    o1_line = ax.plot(x[0], y[0], label='object1 trace')[0]           # object 1
    cartesian_measurements = []
    cartesian_measurements_scatter = ax.scatter([], [], marker='x', color='green', s=5, label='cartesian_measurements')
    polar_measurements = []
    polar_measurements_scatter = ax.scatter([], [], marker='x', color='pink', s=5, label='polar_measurements')
    predictions = []
    predictions_scatter = ax.scatter([], [], marker='x', color='red', s=5, label='prediction')
    ax.set(xlim=[min_x,max_x], ylim=[min_y,max_y], xlabel='x', ylabel='y')
    ax.legend()

    # Update function - does all the calculations
    def update(frame):
        # Object movement
        o1_x_plot = x[:frame]                      # all values until 
        o1_y_plot = y[:frame]                      # current frame
        o1_line.set_xdata(o1_x_plot)
        o1_line.set_ydata(o1_y_plot)

        sensor_data = set()
        for sensor in sensors:
            if frame % sensor.time_between_measurements == 0:
                current_state = x_k[frame,:]
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
                if frame > 0:
                    # Prediction
                    print(frame)
                    old_measurement = np.array((cartesian_measurements[int(frame/sensor.time_between_measurements-1)]))
                    print(old_measurement, np.shape(old_measurement))
                    prediction = sensor.predict()
                    predictions.append(prediction)
                    predictions_scatter.set_offsets(np.array(predictions))
                    sensor_data.add(predictions_scatter)
        return o1_line, s1_pos, sensor_data

    anim = animation.FuncAnimation(fig=fig, func=update, frames=FRAMES, interval=MS_PER_PLOT)
    plt.show()

if __name__ == '__main__': main()