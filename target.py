from . import movement_function as mv
import numpy as np

class Target():
    movement: mv.MovementPath                                       # sth like x = r[:,0] = A * np.sin(omega*t)                    # x(t)
                                                                    #          y = r[:,1] = A * np.sin(2*omega*t)                  # y(t)
    speed: float
    accelaration: np.ndarray
    starting_position_x: float
    starting_position_y: float

    name: str
    color: str

    def __init__(self, movement, color):
        self.movement = movement                                    # some movement function
        self.speed = 1
        self.color = color

    def get_position(current_time):
        # calculate position based on movement_function and 'current_time' (aka t)
        pass