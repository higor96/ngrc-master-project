import numpy as np
from typing import Any

class DoublePendulum:
    """
    Class representing a double pendulum system.
    
    This class provides a method to compute the time derivatives of the system's state,
    which can be used with an ODE solver (e.g., solve_ivp from SciPy) for numerical integration.
    """
    
    @staticmethod
    def double_pendulum_solve_ivp(t: float, state: np.ndarray, L1: float, L2: float, M1: float, M2: float, G: float) -> np.ndarray:
        """
        Compute the time derivatives of the state for a double pendulum system.
        
        The state vector is defined as:
            state[0] : angle of the first pendulum (theta1)
            state[1] : angular velocity of the first pendulum (omega1)
            state[2] : angle of the second pendulum (theta2)
            state[3] : angular velocity of the second pendulum (omega2)
        
        Parameters:
            t (float): Current time (required by ODE solvers, though not used directly in the equations).
            state (np.ndarray): Current state vector [theta1, omega1, theta2, omega2].
            L1 (float): Length of the first pendulum.
            L2 (float): Length of the second pendulum.
            M1 (float): Mass of the first pendulum.
            M2 (float): Mass of the second pendulum.
            G (float): Acceleration due to gravity.
        
        Returns:
            np.ndarray: Derivative of the state vector [d(theta1)/dt, d(omega1)/dt, d(theta2)/dt, d(omega2)/dt].
        """
        # Initialize the derivative vector with zeros (same shape as state)
        dydx = np.zeros_like(state)
        
        # The derivative of theta1 is omega1
        dydx[0] = state[1]
        
        # Calculate the difference between the two pendulum angles
        delta = state[2] - state[0]
        
        # Denominator for the first pendulum's angular acceleration
        den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta)**2
        
        # Compute the angular acceleration of the first pendulum (omega1 derivative)
        dydx[1] = ((M2 * L1 * state[1]**2 * np.sin(delta) * np.cos(delta)
                    + M2 * G * np.sin(state[2]) * np.cos(delta)
                    + M2 * L2 * state[3]**2 * np.sin(delta)
                    - (M1 + M2) * G * np.sin(state[0]))
                   / den1)
        
        # The derivative of theta2 is omega2
        dydx[2] = state[3]
        
        # Denominator for the second pendulum's angular acceleration (scaled from den1)
        den2 = (L2 / L1) * den1
        
        # Compute the angular acceleration of the second pendulum (omega2 derivative)
        dydx[3] = ((- M2 * L2 * state[3]**2 * np.sin(delta) * np.cos(delta)
                    + (M1 + M2) * G * np.sin(state[0]) * np.cos(delta)
                    - (M1 + M2) * L1 * state[1]**2 * np.sin(delta)
                    - (M1 + M2) * G * np.sin(state[2]))
                   / den2)
        
        return dydx
