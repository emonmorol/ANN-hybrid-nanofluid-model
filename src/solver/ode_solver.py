

import numpy as np
from scipy.integrate import solve_bvp
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class HybridNanofluidSolver:
    def __init__(self, params: Dict[str, float]):
        self.M = params.get('M', 1.0)
        self.Nr = params.get('Nr', 0.5)
        self.Nh = params.get('Nh', 0.5)
        self.lam = params.get('lam', 1.0)
        self.beta = params.get('beta', 0.1)
        self.Pr = params.get('Pr', 6.2)
        self.n = params.get('n', 1.0)
        self.Tr = params.get('Tr', 1.5)
        self.As = params.get('As', 1.0)

        # Nanofluid property ratios (typical values for hybrid nanofluids)
        self.nu_ratio = params.get('nu_ratio', 1.05)
        self.kappa_ratio = params.get('kappa_ratio', 1.15)
        self.sigma_ratio = params.get('sigma_ratio', 1.10)
        self.rho_ratio = params.get('rho_ratio', 1.03)
        
        self.eta_max = params.get('eta_max', 10.0)
        self.n_points = params.get('n_points', 400)
        
    def ode_system(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:

        f, fp, fpp, theta, thetap = y
        
        # Momentum equation (Eq. 8): solve for f'''
        # ν_hnf/ν_f * f''' + f*f'' + (2n)/(n+1)*(1 - f'^2) - 2/(n+1) * (σ_hnf/σ_f * ρ_f/ρ_hnf * M) * (f' - 1) = 0
        
        term1 = f * fpp
        term2 = (2.0 * self.n) / (self.n + 1.0) * (1.0 - fp**2)
        term3 = -2.0 / (self.n + 1.0) * (self.sigma_ratio / self.rho_ratio * self.M) * (fp - 1.0)
        
        fppp = -(term1 + term2 + term3) / self.nu_ratio
        
        # Energy equation (Eq. 9): solve for θ''
        # (κ_hnf/κ_f + Nr/(1 + (Tr-1)*θ)^3) * θ'' + Pr*As*(f*θ' - 2(2n-1)/(n+1)*f'*θ) 
        # + 3*Nr*(Tr-1)/(1 + (Tr-1)*θ)^2 * (θ')^2 = 0
        
        theta_term = 1.0 + (self.Tr - 1.0) * theta
        theta_term = np.maximum(theta_term, 0.01)  # Prevent division by zero
        
        coeff_theta_pp = self.kappa_ratio + self.Nr / (theta_term**3)
        
        term1_energy = self.Pr * self.As * (f * thetap - 2.0 * (2.0 * self.n - 1.0) / (self.n + 1.0) * fp * theta)
        term2_energy = 3.0 * self.Nr * (self.Tr - 1.0) / (theta_term**2) * (thetap**2)
        
        thetapp = -(term1_energy + term2_energy) / coeff_theta_pp
        
        return np.vstack([fp, fpp, fppp, thetap, thetapp])
    
    def boundary_conditions(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:

        bc = np.zeros(5)
        # Boundary conditions at η = 0
        bc[0] = ya[0]  # f(0) = 0
        bc[1] = ya[1] - (self.lam + self.beta * ya[2])  # f'(0) = lam + β*f''(0)
        bc[2] = ya[4] + self.Nh * (1.0 - ya[3])  # θ'(0) = -Nh*(1 - θ(0))
        
        # Boundary conditions at η → ∞
        bc[3] = yb[1] - 1.0  # f'(∞) = 1
        bc[4] = yb[3]  # θ(∞) = 0
        
        return bc
    
    def solve(self, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        # Create initial grid
        eta = np.linspace(0, self.eta_max, self.n_points)
        
        # Initial guess (smart initialization)
        f_init = eta - np.exp(-eta)
        fp_init = 1.0 - np.exp(-eta)
        fpp_init = np.exp(-eta)
        theta_init = np.exp(-eta)
        thetap_init = -np.exp(-eta)
        
        y_init = np.vstack([f_init, fp_init, fpp_init, theta_init, thetap_init])
        
        # Solve BVP
        try:
            sol = solve_bvp(self.ode_system, self.boundary_conditions, eta, y_init, 
                          max_nodes=5000, tol=1e-6, verbose=0)
            
            if sol.success:
                if verbose:
                    print(f"✓ Solution converged. RMS residual: {sol.rms_residuals.max():.2e}")
                return sol.x, sol.y
            else:
                if verbose:
                    print(f"✗ Solution failed: {sol.message}")
                return None, None
                
        except Exception as e:
            if verbose:
                print(f"✗ Solver error: {str(e)}")
            return None, None
    
    def compute_derivatives(self, eta: np.ndarray, solution: np.ndarray) -> Dict[str, np.ndarray]:

        return {
            'eta': eta,
            'f': solution[0],
            'fp': solution[1],
            'fpp': solution[2],
            'theta': solution[3],
            'thetap': solution[4]
        }
    
    def compute_engineering_quantities(self, solution: np.ndarray) -> Dict[str, float]:

        fpp_0 = solution[2][0]  # f''(0)
        thetap_0 = solution[4][0]  # θ'(0)
        
        Cf = fpp_0
        Nu = -thetap_0
        
        return {
            'Cf': Cf,
            'Nu': Nu,
            'fpp_0': fpp_0,
            'thetap_0': thetap_0
        }


def test_solver():

    print("Testing Hybrid Nanofluid ODE Solver")
    print("=" * 50)
    
    params = {
        'M': 1.0,
        'Nr': 0.5,
        'Nh': 0.5,
        'lam': 1.0,
        'beta': 0.1,
        'Pr': 6.2,
        'n': 1.0,
        'Tr': 1.5,
        'As': 1.0,
        'eta_max': 10.0,
        'n_points': 400
    }
    
    solver = HybridNanofluidSolver(params)
    eta, solution = solver.solve(verbose=True)
    
    if solution is not None:
        results = solver.compute_derivatives(eta, solution)
        eng_quantities = solver.compute_engineering_quantities(solution)
        
        print(f"\nEngineering Quantities:")
        print(f"  Skin Friction Cf = {eng_quantities['Cf']:.6f}")
        print(f"  Nusselt Number Nu = {eng_quantities['Nu']:.6f}")
        
        print(f"\nBoundary Values:")
        print(f"  f(0) = {results['f'][0]:.6f}")
        print(f"  f'(0) = {results['fp'][0]:.6f}")
        print(f"  f''(0) = {results['fpp'][0]:.6f}")
        print(f"  θ(0) = {results['theta'][0]:.6f}")
        print(f"  θ'(0) = {results['thetap'][0]:.6f}")
        
        print(f"\nAsymptotic Values:")
        print(f"  f'(∞) = {results['fp'][-1]:.6f}")
        print(f"  θ(∞) = {results['theta'][-1]:.6f}")
    else:
        print("Solver failed!")


if __name__ == "__main__":
    test_solver()
