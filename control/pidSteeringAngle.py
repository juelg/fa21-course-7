"""Implements a PID system to compute the optimal steering angle given information about the street middle. 

`cp`, `ci` and `cd` need to be estimated/choosen empirically.

`steeringAngleAlpha` should return the steering angle expected by the car.
"""

import numpy as np

def steeringAnglePIDStep(phi: float, dt: float, dt_1: float, d_total: float, cp: float, ci: float, cd: float) -> float:
    """Computes the new steering angle given the previous distances to the lane middle
    and the previous steering angle.

    Args:
        phi (float): Previous steering angle.
        dt (float): Current distance to road middle.
        dt_1 (float): Last distance to road middle.
        d_total(float): Cumulative Error

        cp (float): Constant for P-controller
        ci (float): Constant for I-controller
        cd (float): Constant for D-controller

    Returns:
        float: New steering angle
    """
    phi_new = min(max(phi + cp*dt + cd*(dt-dt_1) + ci*d_total, -np.pi/4), np.pi/4)
    return phi_new

errors = [0.0, 0.0]
phis = [0.0, 0.0]
def steeringAnglePhi(trajectoryParams: np.array, img_width: int, img_height: int, cp: float, ci: float, cd: float) -> float:
    """Computes the steering angle given a street middle curve.

    However, it only uses the distance at the bottom of our view

    Args:
        trajectoryParams (np.array): parameters for a parabola
        img_width (int): width of image
        img_height (int): height of image
        cp (float): constant for p-controller
        ci (float): constant for i-controller
        cd (float): constant for d-controller

    Returns:
        float: phi=alpha-90
    """
    # Compute Street Middle from Trajectory Params
    y_max = img_height
    middleOfStreet = np.sum(trajectoryParams*np.array([y_max**2, y_max, 1]))

    # Offset if camera introduces a systematic bias
    bias = 57
    errors.append(middleOfStreet-img_width/2-bias)

    # Compute new steering angle
    phi = steeringAnglePIDStep(phis[-1], errors[-1], errors[-2], np.sum(errors), cp, ci, cd)
    phis.append(phi)

    # Prevent Overflow
    if len(errors) > 10:
        del errors[0]
        del phis[0]

    return phi

def steeringAngleAlpha(trajectoryParams: np.array, img_width: int, img_height: int, cp: float, ci: float, cd: float) -> float:
    """Computes the steering angle `alpha`

    Args:
        see `steeringAnglePhi`

    Returns:
        float: `alpha` in radiants
    """
    return steeringAnglePhi(trajectoryParams, img_width, img_height, cp, ci, cd)+np.pi/2

if __name__ == "__main__":
    for i in range(5):
        steeringAnglePhi([0.0, 0.1, 0.0], 640, 480, 0.5, 0.05, 0.4)

    print(phis)
    print(errors)