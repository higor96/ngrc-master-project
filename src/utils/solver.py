import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Any

def simulate_dynamic_system(
    derivs_dynamic: Callable[[float, np.ndarray, Any], np.ndarray],
    initial_state: np.ndarray,
    params: Tuple[Any, ...],
    dt: float,
    max_time: float,
    solve_ivp_method: str = 'RK23'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a dynamic system using an ODE solver.

    This function integrates the system's differential equations using SciPy's solve_ivp.
    It returns the solution array and corresponding time points.

    Parameters:
        derivs_dynamic (Callable): A function that computes the time derivatives of the system.
            It must have the signature:
                derivs_dynamic(t: float, state: np.ndarray, *params) -> np.ndarray
        initial_state (np.ndarray): The initial state vector of the system.
        params (tuple): Additional parameters to be passed to the derivative function.
        dt (float): The time step used for evaluation.
        max_time (float): The maximum simulation time.
        solve_ivp_method (str, optional): The integration method to use (default is 'RK23').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - The first element is an array containing the solution values for each state variable.
            - The second element is an array of time points corresponding to the solution.
    """
    # Create an array of time points at which to evaluate the solution
    t_eval = np.linspace(0, max_time, round(max_time / dt) + 1)
    
    # Solve the ODE system using solve_ivp
    sol = solve_ivp(
        derivs_dynamic, 
        (0, max_time), 
        initial_state, 
        t_eval=t_eval, 
        args=params, 
        method=solve_ivp_method
    )
    
    # Return the solution array and time array
    return sol.y, sol.t
