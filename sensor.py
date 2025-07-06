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

    # other matrices
    x: np.ndarray = np.zeros((6, 1)).flatten()
    P: np.ndarray = np.diag([
        1e2,  # pos_x uncertainty
        1e2,  # pos_y uncertainty
        1e1,  # vel_x uncertainty
        1e1,  # vel_y uncertainty
        1e0,  # acc_x uncertainty
        1e0   # acc_y uncertainty
    ])
    # P = np.zeros((6,6))

    W: np.ndarray
    S: np.ndarray
    v: np.ndarray

    sigma_c = 50
    sigma_r = 20
    sigma_phi = 0.2                                                     # degree
    sigma_k = 0.3

    name: str
    color: str
    time_between_measurements: int = 5

    def __init__(self, color, time_between_measure = 5):
        self.position = np.array((0,0))
        # self.set_position()
        self.H = np.array(((1,0,1,0,1,0), (0,1,0,1,0,1)))
        self.R = (self.sigma_c ** 2) * np.eye(2)
        self.F = np.block([[np.eye(2),                  self.time_between_measurements*np.eye(2),   1/2*(self.time_between_measurements**2)*np.eye(2)],\
                           [np.zeros_like(np.eye(2)),   np.eye(2),                                  self.time_between_measurements*np.eye(2)],\
                           [np.zeros_like(np.eye(2)),   np.zeros_like(np.eye(2)),                   np.eye(2)]])
        
        """ dt = self.time_between_measurements
        q = self.sigma_k**2
        G = np.block([
            [0.5*dt**2*np.eye(2)],
            [dt*np.eye(2)],
            [np.eye(2)]
        ])
        self.D = q * G @ G.T """

        self.D = (self.sigma_k**2)*np.block([[1/4*(self.time_between_measurements**4)*np.eye(2), 1/2*(self.time_between_measurements**3)*np.eye(2),   1/2*(self.time_between_measurements**2)*np.eye(2)],\
                                             [1/2*(self.time_between_measurements**3)*np.eye(2), (self.time_between_measurements**2)*np.eye(2),       self.time_between_measurements*np.eye(2)],\
                                                [1/2*(self.time_between_measurements**2)*np.eye(2), self.time_between_measurements*np.eye(2), np.eye(2)]])
        self.color = color
        if time_between_measure is not None:
            self.time_between_measurements = time_between_measure

    def set_position(self):
        self.position = np.array((random.randrange(-10000, 10001, 1), random.randrange(-10000, 10001, 1)))

    def get_position(self):
        return self.position

    def measure_cartesian(self, object_state):
        if self.x is None: self.x = object_state
        # calculate 'measurement' of_cartesian location and return it
        measurement = self.H@object_state + np.random.normal(scale=self.sigma_c, size=2)
        return measurement

    def measure_polar(self, object_state):
        # calculate polar 'measurement'
        object_state = self.H@object_state
        azimuth = np.array((np.linalg.norm((self.position-object_state)), np.arctan2(object_state[1]-self.position[1], object_state[0]-self.position[0]))) + (np.array((self.sigma_r*np.random.normal(), self.sigma_phi*np.random.normal())))       # (y_k - y_s) / (x_k - x_s)
        range = (object_state-self.position)
        return (azimuth, range)

    def predict(self):
        print("currently used state:", self.H@self.x)
        self.x = self.F @ self.x                                            # x[k|k-1]
        self.P = self.F @ self.P @ self.F.T + self.D                        # P[k|k-1]
        return (self.x, self.P)
    
    def filter(self, measurement):
        self.v = measurement - self.H @ self.x                               # v[k|k-1]
        # print("self.v:", self.v)
        self.S = self.H @ self.P @ self.H.T + self.R                        # S[k|k-1]
        # print("self.S:", self.S)
        self.W = self.P @ self.H.T @ np.linalg.inv(self.S)                   # W[k|k-1]
        # print("self.W:", self.W)

        self.P = self.P - self.W @ self.S @ self.W.T
        self.x = self.x + self.W @ self.v                                   # x[k|k]