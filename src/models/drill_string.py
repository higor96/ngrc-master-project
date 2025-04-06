from collections import namedtuple
import numpy as np
from typing import Any

class TorsionalLateralDrillString:
    """
    Class that encapsulates the dynamics of the drill string using a 
    torsional-lateral model as described in the thesis.
    """
    TLDSState = namedtuple("TLDSState", ["phi", "d_phi", "r", "d_r", "theta", "d_theta"])
    TLDSParameters = namedtuple("TLDSParameters", [
        "c_s", "k_s", "tol", "mu", "b_0", "b_1", "b_2", "b_3",
        "c_h", "e", "R_co", "I_a", "I_m", "m_t", "k_t", "c_t",
        "Omega", "L_c", "W_ob", "V_ref_inv", "E"
    ])

    def __init__(self, Omega: float, W_ob: float) -> None:
        """
        Initialize the drill string model.
        
        :param Omega: Angular speed (in rpm) at the top drive.
        :param W_ob: Weight on bit (in kN).
        """
        # Convert Omega from rpm to rad/s
        RPM = 2 * np.pi / 60
        # Set default parameters (convert Omega to rad/s and W_ob to N)
        self.tlds_parameters = self.tlds_default_parameters(Omega=Omega * RPM, W_ob=W_ob * 1000)
        # Set initial state: [phi, d_phi, r, d_r, theta, d_theta]
        self.initial_tlds_state = TorsionalLateralDrillString.TLDSState(0, 0, 1e-4, 0, 0, 0)

    def f_n(self, u: TLDSState) -> float:
        """
        Calculate the normal force based on the current state.
        
        :param u: Current state of the drill string.
        :return: Normal force.
        """
        p = self.tlds_parameters
        if u.r - p.tol > 0:
            return p.k_s * (u.r - p.tol) + p.c_s * u.d_r
        return 0

    def f_fat(self, u: TLDSState, f_n: float) -> float:
        """
        Calculate the frictional force.
        
        :param u: Current state of the drill string.
        :param f_n: Normal force.
        :return: Frictional force.
        """
        p = self.tlds_parameters
        return p.mu * np.tanh((u.d_phi * p.R_co + u.r * u.d_theta) * p.V_ref_inv) * f_n

    def t_bit(self, u: TLDSState) -> float:
        """
        Compute the bit torque.
        
        :param u: Current state of the drill string.
        :return: Bit torque.
        """
        p = self.tlds_parameters
        return p.W_ob * p.b_0 * (np.tanh(p.b_1 * u.d_phi) + p.b_2 * u.d_phi / (1 + p.b_3 * u.d_phi**2))

    def t_lat(self, u: TLDSState, upsilon: float, f_n: float, f_fat: float) -> float:
        """
        Compute the lateral torque.
        
        :param u: Current state of the drill string.
        :param upsilon: Combined lateral velocity.
        :param f_n: Normal force.
        :param f_fat: Frictional force.
        :return: Lateral torque.
        """
        p = self.tlds_parameters
        return (-f_n * p.e * np.sin(u.phi - u.theta)
                - f_fat * (p.R_co - p.e * np.cos(u.phi - u.theta))
                - p.c_h * upsilon * (u.d_r * p.e * np.sin(u.phi - u.theta)
                                     - u.r * u.d_theta * p.e * np.cos(u.phi - u.theta)))

    def torsional(self, u: TLDSState, t: float, t_bit: float, t_lat: float) -> float:
        """
        Compute the torsional component.
        
        :param u: Current state of the drill string.
        :param t: Current time.
        :param t_bit: Bit torque.
        :param t_lat: Lateral torque.
        :return: Torsional acceleration.
        """
        p = self.tlds_parameters
        return (-t_bit + t_lat) - (p.c_t * (u.d_phi - p.Omega) + p.k_t * (u.phi - p.Omega * t))

    def lateral_r(self, u: TLDSState, upsilon: float, f_n: float, t_bit: float) -> float:
        """
        Compute the radial lateral acceleration.
        
        :param u: Current state of the drill string.
        :param upsilon: Combined lateral velocity.
        :param f_n: Normal force.
        :param t_bit: Bit torque.
        :return: Radial component of lateral acceleration.
        """
        p = self.tlds_parameters
        k = (p.E * p.I_a * np.pi**4 / (2 * p.L_c**3)
             - t_bit * np.pi**3 / (2 * p.L_c**2)
             - p.W_ob * np.pi**2 / (2 * p.L_c))
        return (p.m_t * p.e * u.d_phi**2 * np.cos(u.phi - u.theta) - f_n) - (
            -p.m_t * u.d_theta**2 * u.r + p.c_h * upsilon * u.d_r + k * u.r)

    def lateral_theta(self, u: TLDSState, upsilon: float, f_fat: float) -> float:
        """
        Compute the angular lateral acceleration.
        
        :param u: Current state of the drill string.
        :param upsilon: Combined lateral velocity.
        :param f_fat: Frictional force.
        :return: Angular component of lateral acceleration.
        """
        p = self.tlds_parameters       
        return (p.m_t * p.e * u.d_phi**2 * np.sin(u.phi - u.theta) - f_fat) - (
            p.m_t * 2 * u.d_r * u.d_theta + p.c_h * upsilon * u.r * u.d_theta)

    def mass_matrix(self, u: TLDSState) -> np.ndarray:
        """
        Build the mass matrix used in the dynamic equations.
        
        :param u: Current state of the drill string.
        :return: Mass matrix as a 2D numpy array.
        """
        p = self.tlds_parameters
        return np.array([
            [p.I_m, 0, 0],
            [-p.m_t * p.e * np.sin(u.phi - u.theta), p.m_t, 0],
            [p.m_t * p.e * np.cos(u.phi - u.theta), 0, p.m_t * u.r],
        ])

    def tlds_rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the right-hand side of the differential equations for the drill string dynamics.
        
        :param state: Current state vector [phi, d_phi, r, d_r, theta, d_theta].
        :param t: Current time.
        :return: Time derivatives of the state vector.
        """
        u = TorsionalLateralDrillString.TLDSState(*state)
        p = self.tlds_parameters
        # Compute combined lateral velocity (magnitude of lateral speed)
        upsilon = np.sqrt(u.d_r**2 + u.d_theta**2 * u.r**2)
        # Calculate forces
        f_n_val = self.f_n(u)
        f_fat_val = self.f_fat(u, f_n_val)
        t_bit_val = self.t_bit(u)
        t_lat_val = self.t_lat(u, upsilon, f_n_val, f_fat_val)
        # Compute dynamic contributions
        torsional_val = self.torsional(u, t, t_bit_val, t_lat_val)
        lateral_r_val = self.lateral_r(u, upsilon, f_n_val, t_bit_val)
        lateral_theta_val = self.lateral_theta(u, upsilon, f_fat_val)
        # Solve the mass matrix equation to obtain accelerations
        mass_mat = self.mass_matrix(u)
        rhs_vector = np.array([torsional_val, lateral_r_val, lateral_theta_val])
        rhs = np.linalg.solve(mass_mat, rhs_vector)
        # Return the derivative vector: [d_phi, torsional acceleration, d_r, lateral r acceleration, d_theta, lateral theta acceleration]
        return np.array([u.d_phi, rhs[0], u.d_r, rhs[1], u.d_theta, rhs[2]])

    def tlds_rhs_solve_ivp(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Wrapper for integration routines (e.g., solve_ivp) to compute the state derivatives.
        
        :param t: Current time.
        :param state: Current state vector.
        :return: Derivative of the state vector.
        """
        return self.tlds_rhs(state, t)

    def tlds_default_parameters(self, Omega: float, W_ob: float, E: float = 220e9, G: float = 85.30e9, e: float = 0.014, 
                                 rho: float = 7800, rho_f: float = 1500, C_A: float = 1.7, C_D: float = 1.0,
                                 beta_1: float = 0.35, beta_2: float = 0.06, k_s: float = 10e9, mu: float = 0.35, 
                                 g: float = 9.81, V_ref: float = 1e-4, kappa: float = 0.0046, eta: float = 4.66,
                                 b_0: float = 0.0239, b_1: float = 1.910, b_2: float = 8.500, b_3: float = 5.470,
                                 D_po: float = 0.140, D_pi: float = 0.119, L_BHA1: float = 171.30,
                                 D_BHA1o: float = 0.140, D_BHA1i: float = 0.076, L_BHA2: float = 294.90,
                                 D_BHA2o: float = 0.171, D_BHA2i: float = 0.071, L_c: float = 8.550,
                                 D_co: float = 0.171, D_ci: float = 0.071, D_bit: float = 0.216, L_p: float = 4733.6,
                                 c_s: float = 1000.0, V_ref_inv: float = 1e8) -> Any:
        """
        Calculate and return the default parameters for the drill string model.
        
        :param Omega: Angular speed in rad/s.
        :param W_ob: Weight on bit in N.
        :param E: Young's modulus in Pa.
        :param G: Shear modulus in Pa.
        :param e: Eccentricity in m.
        :param rho: Drill pipe density in kg/m³.
        :param rho_f: Fluid density in kg/m³.
        :param C_A: Added mass coefficient.
        :param C_D: Drag coefficient.
        :param beta_1: Proportional damping coefficient (s⁻¹).
        :param beta_2: Proportional damping coefficient (s).
        :param k_s: Contact stiffness in N/m.
        :param mu: Wall friction coefficient.
        :param g: Gravitational acceleration in m/s².
        :param V_ref: Friction constant in m/s.
        :param kappa: Gamma PDF shape parameter.
        :param eta: Gamma PDF scale parameter.
        :param b_0: Bit torque model constant.
        :param b_1: Bit torque model constant.
        :param b_2: Bit torque model constant.
        :param b_3: Bit torque model constant.
        :param D_po: Outer diameter of the drill pipe in m.
        :param D_pi: Inner diameter of the drill pipe in m.
        :param L_BHA1: Length of BHA Section 1 in m.
        :param D_BHA1o: Outer diameter of BHA Section 1 in m.
        :param D_BHA1i: Inner diameter of BHA Section 1 in m.
        :param L_BHA2: Length of BHA Section 2 in m.
        :param D_BHA2o: Outer diameter of BHA Section 2 in m.
        :param D_BHA2i: Inner diameter of BHA Section 2 in m.
        :param L_c: Length between stabilizers in m.
        :param D_co: Outer diameter between stabilizers in m.
        :param D_ci: Inner diameter between stabilizers in m.
        :param D_bit: Borehole wall diameter in m.
        :param L_p: Length of the drill pipe in m.
        :param c_s: Contact damping coefficient.
        :param V_ref_inv: Inverse of V_ref.
        :return: TLDSParameters namedtuple with all default parameters.
        """
        R_co = D_co / 2
        D_wall = D_bit

        def I_m_x(D_o: float, D_i: float, L: float) -> float:
            """
            Calculate the mass moment of inertia for a pipe segment.
            
            :param D_o: Outer diameter.
            :param D_i: Inner diameter.
            :param L: Length of the segment.
            :return: Mass moment of inertia.
            """
            vol = np.pi * (D_o**2 - D_i**2) * L / 4
            return rho * vol * (D_o**2 + D_i**2) / 8

        tol = (D_wall - D_co) / 2
        c_h = 2 * rho_f * C_D * D_co * L_c / (3 * np.pi)
        m = np.pi * rho * (D_co**2 - D_ci**2) * L_c / 8
        m_f = np.pi * rho_f * (D_ci**2 + C_A * D_co**2) * L_c / 8
        m_t = m + m_f
        I_a = np.pi * (D_co**4 - D_ci**4) / 64
        J_p = np.pi * (D_po**4 - D_pi**4) / 32
        k_t = G * J_p / L_p
        I_m_p = I_m_x(D_po, D_pi, L_p)
        I_m_BHA1 = I_m_x(D_BHA1o, D_BHA1i, L_BHA1)
        I_m_BHA2 = I_m_x(D_BHA2o, D_BHA2i, L_BHA2)
        I_m = I_m_p / 3 + I_m_BHA1 + I_m_BHA2
        c_t = beta_1 * I_m + beta_2 * k_t

        return TorsionalLateralDrillString.TLDSParameters(
            c_s=c_s, k_s=k_s, tol=tol, mu=mu, b_0=b_0, b_1=b_1, b_2=b_2, b_3=b_3,
            c_h=c_h, e=e, R_co=R_co, I_a=I_a, I_m=I_m, m_t=m_t, k_t=k_t,
            c_t=c_t, Omega=Omega, L_c=L_c, W_ob=W_ob, V_ref_inv=V_ref_inv, E=E
        )
