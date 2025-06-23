import numpy as np
import random

class Sensor():
    # cartesian coordinates
    position = np.ndarray((2,1))

    # sensor model
    H: np.ndarray = np.ndarray((2,2))                                   # accuracy
    R: np.ndarray = np.ndarray((2,2))                                   # uncertainty
    # dynamics model
    F: np.ndarray = np.ndarray((2,2))
    D: np.ndarray = np.ndarray((2,2))

    sigma_c = 50
    sigma_r = 20
    sigma_phi = 0.2                                                     # degree

    name: str
    color: str
    time_between_measurements: int = 5

    def __init__(self, color, time_between_measure = None):
        self.set_position()
        self.H = np.zeros((2, 6))                                       # H = (I,O,O)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.R = self.sigma_c * np.array((np.random.normal(), np.random.normal()))
        self.color = color
        if time_between_measure is not None:
            self.time_between_measurements = time_between_measure

    def set_position(self):
        self.position = np.array((random.randrange(-10000, 10001, 1), random.randrange(-10000, 10001, 1)))

    def get_position(self):
        return self.position

    def measure_cartesian(self, object_state):
        # calculate 'measurement' of_cartesian location and return it
        measurement = self.H@object_state + np.random.normal(scale=self.sigma_c, size=2)
        return measurement

    def measure_polar(self, object_state):
        # calculate polar 'measurement'
        object_state = self.H@object_state
        azimuth = np.array((np.linalg.norm((self.position-object_state)), np.arctan2(object_state[1]-self.position[1], object_state[0]-self.position[0]))) + (np.array((self.sigma_r*np.random.normal(), self.sigma_phi*np.random.normal())))       # (y_k - y_s) / (x_k - x_s)
        range = (object_state-self.position)
        return (azimuth, range)