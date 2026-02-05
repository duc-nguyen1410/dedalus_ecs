"""
    python3 continuation.py
"""

import subprocess
import numpy as np
import os
import h5py
from enum import Enum
class Mu(Enum):
    Lx = "Lx"
    Lz = "Lz"
    Ra = "Ra"
    Pr = "Pr"
    tau = "tau"
    Rrho = "Rrho"

MU_REF = 1
DS_MIN = 1e-3
DS_MAX = 0.1

GUESS_ERR_MIN = 0.1   # acceptable *lower* bound for guesserr
GUESS_ERR_MAX = 100.0   # acceptable *upper* bound for guesserr
def L2dist(x1, x2):
    return np.linalg.norm(x1 - x2)
def update_3list(lst, new_val):
    lst[0] = lst[1]
    lst[1] = lst[2]
    lst[2] = new_val
def get_state(input_filename):
    # Load initial guess from Dedalus simulation data at (nearest) time fixed_t
    input_file = h5py.File(input_filename, 'r')
    t = np.array(input_file.get('/t')) # temperature in grid space
    s = np.array(input_file.get('/s')) # salinity in grid space
    u = np.array(input_file.get('/u')) # x-velocity in grid space
    w = np.array(input_file.get('/w')) # y-velocity in grid space
    Nx = t.shape[0]
    Nz = t.shape[1]
    # Flatten and concatenate into a single state vector
    return Nx,Nz,np.concatenate([t.flatten(), s.flatten(), u.flatten(), w.flatten()])

def quadraticInterpolate(xn, mun, mu, eps=1e-12):
    """
    Python version of Channelflow's quadraticInterpolate for arrays of vectors.
    xn: list of 3 numpy arrays [f0, f1, f2]
    mun: list/array of 3 floats [mu0, mu1, mu2]
    mu: float, evaluation point
    eps: float, tolerance for constancy
    Returns: numpy array of shape (N,)
    """
    # Convert input into (3, N)
    xn = [np.atleast_1d(x) for x in xn]
    xn = np.vstack(xn)   # shape (3, N)
    mun = np.asarray(mun) # shape (3,)

    if xn.shape[0] != 3 or mun.shape[0] != 3:
        raise ValueError("xn must have shape (3, N), mun must have length 3")
    
    N = xn.shape[1] # get element count of state vector [t,s,u,v]
    if N == 1:
        fn = xn[:, 0]  # length-3 vector
        if np.all(np.abs(fn - fn[0]) < eps):
            return float(fn[0])
        coeffs = np.polyfit(mun, fn, 2)
        return float(np.polyval(coeffs, mu))

    x = np.empty(N) # create output vector
    # get eat element of the vectors in xn, mun
    # then predict element xnew[i] at mu
    for i in range(N):
        fn = xn[:,i] # get ith element of each vector in xn --> fn = [f0[i], f1[i], f2[i]]
        if np.all(np.abs(fn - fn[0]) < eps): # quick check: nearly constant
            x[i] = fn[0]
        else:
            coeffs = np.polyfit(mun, fn, 2)
            x[i] = np.polyval(coeffs, mu)
    return x

def secant_predict(mu_prev, mu_curr, norm_prev, norm_curr, ds):
    # tangent vector
    dmu = mu_curr - mu_prev
    dnorm = norm_curr - norm_prev
    # normalize tangent length
    L = (dmu**2 + dnorm**2)**0.5
    mu_pred = mu_curr + ds * dmu / L
    norm_pred = norm_curr + ds * dnorm / L
    return mu_pred, norm_pred
def quad_predict_fold(mu_vals, norm_vals):
    """
    Fit quadratic norm(mu) and return vertex (fold point).
    mu_vals: [mu0, mu1, mu2]
    norm_vals: [norm0, norm1, norm2]
    Returns (mu_star, norm_star, coeffs)
    """
    coeffs = np.polyfit(mu_vals, norm_vals, 2)  # [a, b, c] for a mu^2 + b mu + c
    a, b, c = coeffs
    if abs(a) < 1e-12:
        # nearly linear, fall back to secant
        mu_star = mu_vals[-1]
        norm_star = norm_vals[-1]
    else:
        mu_star = -b / (2*a)         # fold μ
        norm_star = np.polyval(coeffs, mu_star)
    return mu_star, norm_star, coeffs

class ECSContinuation:
    def __init__(self,
                 ecs_name="eqb",
                 xrel = False, ax = 1e-12,
                 zrel = False, az = 1e-12,
                 mu_name="Lx",
                 folder="./", 
                 guess="initial_guess.h5", 
                 kx=None, Lx = 1.0, kz = None, Lz=1.0, 
                 Ra=1e5, Pr=7.0, Rrho=40.0, Lambda=2.0, tau=0.01, 
                 T=20,dt=0.0001,adjust_dt=False,
                 eigen = False,
                 symm=False,symmfile="",
                 n_procs=16):
        
        self.ecs = ecs_name
        self.xrel, self.ax = xrel, ax
        self.zrel, self.az = zrel, az
        self.mu_name = mu_name
        self.kx = kx if kx!=None else None
        self.kz = kz if kz!=None else None
        self.Lx = 2*np.pi/self.kx if kx!=None else Lx
        self.Lz = 2*np.pi/self.kz if kz!=None else Lz
        if mu_name=="kx":
            self.mu = kx
        elif mu_name=="Lx":
            self.mu = Lx
        elif mu_name=="kz":
            self.mu = kz
        elif mu_name=="Lz":
            self.mu = Lz
        elif mu_name=="Ra":
            self.mu = Ra
        elif mu_name=="Pr": 
            self.mu = Pr
        elif mu_name=="Rrho": 
            self.mu = Rrho
        elif mu_name=="Lambda": 
            self.mu = Lambda
        elif mu_name=="tau":
            self.mu = tau
        else:
            None                        
        self.folder = folder
        self.isearch = 0
        self.odir = self.folder+"search-"+str(self.isearch)+"/"
        self.Nx = None
        self.Nz = None
        self.Ra = Ra
        self.Pr = Pr
        self.Rrho = Rrho
        self.Lambda = Lambda
        self.tau = tau
        self.inputpath = guess
        self.T = T
        self.dt = dt
        self.adjust_dt = adjust_dt
        self.eigen = eigen
        self.symm = symm
        self.symmfile = symmfile
        self.n_procs = n_procs
        self.solutions = []  # store (mu, solution_norm) pairs

        print(f"Initialized continuation with mu={self.mu_name}, folder={self.folder}, guess={self.inputpath}")
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        # load initial guess
        if not os.path.exists(self.inputpath):
            raise FileNotFoundError(f"Initial guess file {self.inputpath} not found.")
        else:
            print(f"Loading initial guess from {self.inputpath}")
            self.Nx,self.Nz,self.solutions = get_state(self.inputpath)

        print('---------------------------------')
        print('--- Initial Parameters ----------')
        print('---------------------------------')
        print('Ra = ' + str(self.Ra))
        print('Pr = ' + str(self.Pr))
        print('Rrho = ' + str(self.Rrho))
        print('Lambda = ' + str(self.Lambda))
        print('tau = ' + str(self.tau))
        print('Nx = ' + str(self.Nx))
        print('Nz = ' + str(self.Nz))
        print('Lz = ' + str(self.Lz))
        print('Lx = ' + str(self.Lx))
        print('---------------------------------')

    def save_state(self, x, filename):
        h5f = h5py.File(filename, 'w')
        #------------ Simulation parameters -------------------- #
        # h5f.create_dataset('/params/Ra', data = self.Ra)
        # h5f.create_dataset('/params/Pr', data = self.Pr)
        # h5f.create_dataset('/params/Rrho', data = self.Rrho)
        # h5f.create_dataset('/params/tau', data = self.tau)
        # h5f.create_dataset('/params/kx', data = self.kx)
        # h5f.create_dataset('/params/kz', data = self.kz)
        # h5f.create_dataset('/params/Nx', data = self.Nx)
        # h5f.create_dataset('/params/Nz', data = self.Nz)
        # h5f.create_dataset('/params/Lx', data = self.Lx)
        # h5f.create_dataset('/params/Lz', data = self.Lz)
        #------------ Grid space data -------------------- #
        h5f.create_dataset('t', data = x[0:self.Nx*self.Nz].reshape((self.Nx,self.Nz)))
        h5f.create_dataset('s', data = x[self.Nx*self.Nz:2*self.Nx*self.Nz].reshape((self.Nx,self.Nz)))
        h5f.create_dataset('u', data = x[2*self.Nx*self.Nz:3*self.Nx*self.Nz].reshape((self.Nx,self.Nz)))
        h5f.create_dataset('w', data = x[3*self.Nx*self.Nz:4*self.Nx*self.Nz].reshape((self.Nx,self.Nz)))
        #--------------------------------------------------------------------- #
        h5f.close()

    def run_ecs(self, x0):
        """Run your existing ECS code for a given mu"""
        if not os.path.exists(self.folder+"temp_eval/"):
            os.mkdir(self.folder+"temp_eval/")
        self.save_state(x0, self.folder+"temp_eval/guess.h5")
        self.inputpath = self.folder+"temp_eval/guess.h5"


        cmd = ["mpiexec", "-n", str(self.n_procs), "python3", "find_ecs.py"]
        def add_arg(key, value):
            cmd.extend([key, str(value)])
        add_arg("--ecs", self.ecs)
        if self.xrel:
            cmd.append("--xrel")
            add_arg("--ax", self.ax)
        if self.zrel:
            cmd.append("--zrel")
            add_arg("--az", self.az)
        add_arg("--odir", self.odir)
        add_arg("--kx" if self.kx is not None else "--Lx",
                self.kx if self.kx is not None else self.Lx)
        add_arg("--kz" if self.kz is not None else "--Lz",
                self.kz if self.kz is not None else self.Lz)
        add_arg("--Ra", self.Ra)
        add_arg("--Pr", self.Pr)
        add_arg("--Rrho", self.Rrho)
        add_arg("--Lambda", self.Lambda)
        add_arg("--tau", self.tau)
        add_arg("--T", self.T)
        if self.adjust_dt:
            cmd.append(f"--adjust_dt")
        add_arg("--dt", self.dt)
        if self.eigen:
            cmd.append(f"--eigen")
        if self.symm:
            cmd.append(f"--symm")
            add_arg("--symmfile", self.symmfile)
        add_arg("--input", self.inputpath)


        if not os.path.exists(self.odir):
            os.mkdir(self.odir)
        
        if self.isearch > -1:
            print("Running:", " ".join(cmd))
            with open(self.odir+"log.txt", "w") as f:
                result = subprocess.run(cmd, stdout=f, text=True)
                if result.returncode != 0:
                    print(f"ECS solver failed at {self.mu_name} = ", self.print_mu())
                    print(result.stderr)
                    return None

        # load tol to check convergence
        tol_file = self.odir + "tol.txt"
        if os.path.exists(tol_file):
            with open(tol_file, "r") as f:
                tol = float(f.read().strip())
        else:
            tol = None

        # load norm of the solution
        norm_file = self.odir + "norm.txt"
        if os.path.exists(norm_file):
            with open(norm_file, "r") as f:
                norm = float(f.read().strip())
        else:
            norm = None

        # load solution
        _,_,soln = get_state(self.odir+"solution.h5")

        if tol < 1e-10:
            print(f"ECS converged for {self.mu_name} = {self.print_mu()} with tol = {tol}, L2(u) = {norm}")
        else:
            print(f"ECS did not converge for {self.mu_name} = {self.print_mu()}, final tol = {tol}")
        return soln, tol, norm
        
    def try_eval_guess(self, x0):
        """Evaluate the guess error without running the full ECS solver"""
        if not os.path.exists(self.folder+"temp_eval/"):
            os.mkdir(self.folder+"temp_eval/")
        self.save_state(x0, self.folder+"temp_eval/guess.h5") ### <-----

        cmd = ["mpiexec", "-n", str(self.n_procs), "python3", "find_ecs.py"]
        def add_arg(key, value):
            cmd.extend([key, str(value)])
        # add_arg("--ecs", self.ecs) ### <-----
        if self.xrel:
            cmd.append("--xrel")
            add_arg("--ax", self.ax)
        if self.zrel:
            cmd.append("--zrel")
            add_arg("--az", self.az)
        add_arg("--odir", self.folder+"temp_eval/") ### <-----
        add_arg("--kx" if self.kx is not None else "--Lx",
                self.kx if self.kx is not None else self.Lx)
        add_arg("--kz" if self.kz is not None else "--Lz",
                self.kz if self.kz is not None else self.Lz)
        add_arg("--Ra", self.Ra)
        add_arg("--Pr", self.Pr)
        add_arg("--Rrho", self.Rrho)
        add_arg("--Lambda", self.Lambda)
        add_arg("--tau", self.tau)
        add_arg("--T", self.T)
        if self.adjust_dt:
            cmd.append(f"--adjust_dt")
        add_arg("--dt", self.dt)
        # if self.eigen:
        #     cmd.append(f"--eigen") ### <-----
        if self.symm:
            cmd.append(f"--symm")
            add_arg("--symmfile", self.symmfile)
        add_arg("--input", f"{self.folder}temp_eval/guess.h5") ### <-----

        
        # print("Evaluating guess with:", " ".join(cmd))
        print("Evaluating guess with ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Guess evaluation failed.")
            print(result.stderr)
            return None
        try:
            # load error of the solution
            tol_file = f"{self.folder}temp_eval/" + "tol.txt"
            if os.path.exists(tol_file):
                with open(tol_file, "r") as f:
                    guesserr = float(f.read().strip())
            else:
                guesserr = None
            return guesserr
        except ValueError:
            print("Failed to parse guess error from output.")
            return None
        


    def load_flow_properties(self):
        # If file does not exist, create and write header
        # if not os.path.exists(self.folder+'flow_properties.txt'):
        #     with open(self.folder+'flow_properties.txt', 'w') as f:
        #         f.write("mu, L2u, L2t, L2s, KE, Nu, Sh, dir\n")
        #         f.close()

        # load flow properties and append to a common file
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/flow_properties.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/flow_properties.txt", 'r') as f:
                lines = f.readlines() # read all lines
                last_line = lines[-1].strip() # get last non-empty line
                f.close()

                if not os.path.exists(self.folder+'flow_properties.txt'):
                    title = lines[0].strip()
                    with open(self.folder+'flow_properties.txt', 'w') as f2:
                        f2.write(f"mu, {title}, dir\n")
                        f2.close()

                return last_line
        else:
            return None
    
    
    def load_T(self):
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/T.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/T.txt", 'r') as f:
                self.T = float(f.read().strip())
                f.close()
        else:
            return None
    def load_ax(self):
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/ax.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/ax.txt", 'r') as f:
                self.ax = float(f.read().strip())
                f.close()
        else:
            return None
    def load_az(self):
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/az.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/az.txt", 'r') as f:
                self.az = float(f.read().strip())
                f.close()
        else:
            return None
    def load_return_T(self):
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/T.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/T.txt", 'r') as f:
                T = float(f.read().strip())
                f.close()
                return T
        else:
            return None
    def load_return_ax(self):
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/ax.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/ax.txt", 'r') as f:
                ax = float(f.read().strip())
                f.close()
                return ax
        else:
            return None
    def load_return_az(self):
        if os.path.exists(self.folder+"search-"+str(self.isearch)+"/az.txt"):
            with open(self.folder+"search-"+str(self.isearch)+"/az.txt", 'r') as f:
                az = float(f.read().strip())
                f.close()
                return az
        else:
            return None
        
    def update_isearch(self):
        self.isearch += 1
        self.odir = self.folder+"search-"+str(self.isearch)+"/"
    def update_mu(self, new_mu):
        if self.mu_name == "kx":
            print(f"Updating {self.mu_name} from {self.kx} to {new_mu}")
            self.kx = new_mu
        elif self.mu_name == "kz":
            print(f"Updating {self.mu_name} from {self.kz} to {new_mu}")
            self.kz = new_mu
        elif self.mu_name == "Lx":
            print(f"Updating {self.mu_name} from {self.Lx} to {new_mu}")
            self.Lx = new_mu
        elif self.mu_name == "Lz":
            print(f"Updating {self.mu_name} from {self.Lz} to {new_mu}")
            self.Lz = new_mu
        elif self.mu_name == "Ra":
            print(f"Updating {self.mu_name} from {self.Ra} to {new_mu}")
            self.Ra = new_mu
        elif self.mu_name == "Pr":
            print(f"Updating {self.mu_name} from {self.Pr} to {new_mu}")
            self.Pr = new_mu
        elif self.mu_name == "Rrho":
            print(f"Updating {self.mu_name} from {self.Rrho} to {new_mu}")
            self.Rrho = new_mu
        elif self.mu_name == "Lambda":
            print(f"Updating {self.mu_name} from {self.Lambda} to {new_mu}")
            self.Lambda = new_mu
        elif self.mu_name == "tau":
            print(f"Updating {self.mu_name} from {self.tau} to {new_mu}")
            self.tau = new_mu
        else:
            print(f"Unknown mu_name {self.mu_name}, no update performed.")
    def print_mu(self):
        if self.mu_name == "kx":
            return str(self.kx)
        elif self.mu_name == "kz":
            return str(self.kz)
        elif self.mu_name == "Lx":
            return str(self.Lx)
        elif self.mu_name == "Lz":
            return str(self.Lz)
        elif self.mu_name == "Ra":
            return str(self.Ra)
        elif self.mu_name == "Pr":
            return str(self.Pr)
        elif self.mu_name == "Rrho":
            return str(self.Rrho)
        elif self.mu_name == "Lambda":
            return str(self.Lambda)
        elif self.mu_name == "tau":
            return str(self.tau)
        else:
            return "Unknown mu_name"
        
    def natural_continuation(self, mu_values, restart_file=None):
        """Normal continuation using previous solution as initial guess"""
        
        if os.path.exists(self.folder+'./flow_properties.txt'):
            os.remove(self.folder+'./flow_properties.txt')
        for mu in mu_values:
            self.update_mu(mu)
            print(f"\n>>> Search ecs for {self.mu_name}={self.print_mu()} in folder={self.odir}")
            tol, _ = self.run_ecs()
            with open(self.odir+'mu.txt', 'w') as f:
                f.write(f"{mu}\n")
                f.close()
            if tol < 1e-10:
                # save flow properties
                property = self.load_flow_properties()
                with open(self.folder+'flow_properties.txt', 'a') as f:
                    f.write(f"{str(mu)}, " + property + "\n")
                    f.close()
                # update guess for next iteration
                self.inputpath = self.folder+"search-"+str(self.isearch)+"/solution.h5"

            self.update_isearch()
    
    

    # Example: three observed points (mu, metric)
    # mu_vals = [14.0, 14.6, 15.1]
    # s_vals  = [0.1, 0.12, 0.11]  # e.g. amplitude, norm, etc.
    # s_star, mu_star, coeffs = quad_predict_fold(mu_vals, s_vals)
    # print("Predicted fold at s*=", s_star, " mu*=", mu_star, " coeffs=", coeffs)

    def arc_length_continuation(self, mu_start, dmu, n_steps=1000, mu_target=None, restart_file=None):
        """
        Pseudo-arclength continuation
        ds: step along the solution curve
        n_steps: max number of continuation steps
        """
        
        if os.path.exists(self.folder+'./flow_properties.txt'):
            os.remove(self.folder+'./flow_properties.txt')

        obs = np.empty(3)  # observable (norm)
        s = np.empty(3)    # (mu,obs) arclength parameter
        res = np.empty(3)  # residual (not used here)
        mu = np.empty(3)   # continuous parameter values

        x = [None, None, None]  # solutions at three points
        mu[0], mu[1], mu[2] = mu_start - dmu, mu_start, mu_start + dmu # Initial two points using natural continuation to estimate tangent
        T = [self.T, self.T, self.T] # time period
        ax = [self.ax, self.ax, self.ax] # x-shifting interval per Lx in a T time period
        az = [self.az, self.az, self.az]

        #### find solutions for initial data
        self.update_mu(mu[0])
        print(f"\n>>> Search ecs for {self.mu_name}={self.print_mu()} in folder={self.odir}")
        x[0], res[0], obs[0] = self.run_ecs(self.solutions)
        # tol0, norm0 = 5.164893899961121e-11, 0.7245739950860248
        property = self.load_flow_properties()
        with open(self.folder+'flow_properties.txt', 'a') as f:
            f.write(f"{mu[0]}, " + property + ", search-"+str(self.isearch)+"/" + "\n")
            f.close()
        with open(self.odir+'mu.txt', 'w') as f:
            f.write(f"{mu[0]}\n")
            f.close()
        if self.ecs == "orb":
            T[0] = self.load_return_T()
        if self.xrel:
            ax[0] = self.load_return_ax()
        if self.zrel:
            az[0] = self.load_return_az()

        self.update_mu(mu[1])
        self.update_isearch()
        print(f"\n>>> Search ecs for {self.mu_name}={self.print_mu()} in folder={self.odir}")
        x[1], res[1], obs[1] = self.run_ecs(self.solutions)
        # tol1, norm1 = 9.711306017873108e-11, 0.7255067044872312
        property = self.load_flow_properties()
        with open(self.folder+'flow_properties.txt', 'a') as f:
            f.write(f"{mu[1]}, " + property + ", search-"+str(self.isearch)+"/" + "\n")
            f.close()
        with open(self.odir+'mu.txt', 'w') as f:
            f.write(f"{mu[1]}\n")
            f.close()
        if self.ecs == "orb":
            T[1] = self.load_return_T()
        if self.xrel:
            ax[1] = self.load_return_ax()
        if self.zrel:
            az[1] = self.load_return_az()

        self.update_mu(mu[2])
        self.update_isearch()
        print(f"\n>>> Search ecs for {self.mu_name}={self.print_mu()} in folder={self.odir}")
        x[2], res[2], obs[2] = self.run_ecs(self.solutions)
        # tol2, norm2 = 9.711306017873108e-11, 0.7332914803120617
        property = self.load_flow_properties()
        with open(self.folder+'flow_properties.txt', 'a') as f:
            f.write(f"{mu[2]}, " + property + ", search-"+str(self.isearch)+"/" + "\n")
            f.close()
        with open(self.odir+'mu.txt', 'w') as f:
            f.write(f"{mu[2]}\n")
            f.close()
        if self.ecs == "orb":
            T[2] = self.load_return_T()
        if self.xrel:
            ax[2] = self.load_return_ax()
        if self.zrel:
            az[2] = self.load_return_az()

        
        if res[0] is None or res[0]>1e-10 or res[1] is None or res[1]>1e-10 or res[2] is None or res[2]>1e-10:
            print("Failed to initialize arclength continuation")
            return

        # Form independent variable (arclength coordinate)
        # In Channelflow, s is the arclength parameter, obs is the observable (norm), mu is the parameter
        # Here, we mimic the logic in Python

        # Compute normalization factors
        ds = None  # initial step size along arclength
        obsnorm = abs(obs[1])  # use middle point for normalization
        munorm = abs(MU_REF) if abs(MU_REF) >= 1e-12 else abs(mu[1])
        if abs(munorm) < 1e-12:
            munorm = 1

        s0 = 0.0  # start value for arclength (arbitrary)
        ds0 = 0 # initial guess for step size along arclength (arbitrary)
        print(f"     s0 == {s0}")
        print(f" munorm == {munorm}")
        for i in range(3):
            print(f"  mu[{i}] == {mu[i]}")
        print(f"obsnorm == {obsnorm}")
        for i in range(3):
            print(f" obs[{i}] == {obs[i]}")

        # Compute arclength coordinates s[0], s[1], s[2]
        s = [0, 0, 0]
        # If you want to mimic cflags.arclength, set arclength=True
        arclength = False
        if arclength:
            # Use Euclidean distance in (state, mu) space
            # Here, we only have norm (observable), not full state vector x
            s[0] = s0 - np.sqrt(L2dist(x[0],x[1]) + ((mu[0] - mu[1]) / munorm)**2)
            s[1] = s0
            s[2] = s0 + np.sqrt(L2dist(x[2],x[1]) + ((mu[2] - mu[1]) / munorm)**2)
            ds = ds0 if ds0!=0 else np.sqrt(L2dist(x[2],x[1]) + ((mu[2] - mu[1]) / munorm)**2)
        else:
            # Use only observable and mu
            s[0] = s0 - np.hypot((obs[0] - obs[1]) / obsnorm, (mu[0] - mu[1]) / munorm)
            s[1] = s0
            s[2] = s0 + np.hypot((obs[2] - obs[1]) / obsnorm, (mu[2] - mu[1]) / munorm)
            ds = ds0 if ds0!=0 else np.hypot((obs[2] - obs[1]) / obsnorm, (mu[2] - mu[1]) / munorm)

        for i in range(3):
            print(f"   s[{i}] == {s[i]}")
        print(f"      ds == {ds}")

        # If you had full state vectors, you could extend them by mu as in Channelflow
        # In this code, we only have norm, so we skip that part


        # self.inputpath = self.folder+"search-"+str(self.isearch)+"/solution.h5"

        
        
        # Based on Channelflow 2.0 logic
        Ndsadjust = 6
        prev_search_failed = False
        guesserrtarget = np.sqrt(GUESS_ERR_MIN * GUESS_ERR_MAX);  # aim between error bounds
        guesserr_prev_search = guesserrtarget
        reachedTarget = False

        # continuation
        for step in range(n_steps-3):
            if (mu_target != None and (mu_target - mu[2]) * (mu_target - mu[1]) < 0):
                print(f"Target mu={mu_target}  reached at step {step} with mu={self.print_mu()}")
                reachedTarget = True
                break

            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Continuing previous solutions by quadratic extrapolation in s = (mu,obs)")
            W = 12  # column width for formatting
            print(f"{'i':>8}{'s':>{W}}{'mu':>{W}}")
            for i in range(3):
                print(f"{i:>8}{s[i]:>{W}.6g}{mu[i]:>{W}.6g}")

            ds_reached_bounds = False
            no_more_adjustments = False

            munew = None
            snew = None
            resnew = None
            xnew = None

            Tnew = None
            axnew = None
            aznew = None

            self.update_isearch()

            # Find a guess with error within error bounds, or under it if previous Newton search has failed
            for iadjust in range(Ndsadjust):
                print("------------------------------------------------------")
                print(f"Look for decent initial guess, adjust {iadjust}")
                # 2) desired new arclength
                snew = s[2] + ds

                # 3) predict mu(snew) by interpolating mu as a function of s
                # Use Neville quadratic (your quadratic_interpolate expects xn, fn, x)
                munew = quadraticInterpolate(mu, s, snew)
                
                self.update_mu(munew)
                # predict state vector at predicted mu by interpolating x(mu)
                print(f"calculating error of extrapolated guess for mu = {munew}")
                xnew = quadraticInterpolate(x, s, snew); # here x[i] is state vector (t,s,u,v) of previous 3 solutions
                ### and then use xnew as initial guess for Newton solver

                if self.ecs == "orb":
                    Tnew = quadraticInterpolate(T, s, snew)
                    self.T = Tnew
                if self.xrel:
                    axnew = quadraticInterpolate(ax, s, snew)
                    self.ax = axnew
                if self.zrel:
                    aznew = quadraticInterpolate(az, s, snew)
                    self.az = aznew

                guesserr = self.try_eval_guess(xnew)

                print(f"dsmin == {DS_MIN}")
                print(f"ds    == {ds}")
                print(f"dsmax == {DS_MAX}")
                print(f"guesserrmin == {GUESS_ERR_MIN}")
                print(f"guesserr    == {guesserr}")
                print(f"guesserrmax == {GUESS_ERR_MAX}")
                print(f"guesserr previous  == {guesserr_prev_search}")
                print(f"prev_search_failed == {prev_search_failed}")
                print(f"ds_reached_bounds  == {ds_reached_bounds}")
                print(f"no_more_adjustments == {no_more_adjustments}")

                # 4) predict norm at predicted mu by interpolating norm(mu)
                # norm_pred = quadraticInterpolate(obs, s, munew)
                
                # break 
                # print(f"Extrapolated guess: munew={munew:.6g}, norm_pred={norm_pred:.6g}")
                # print(f"mu_prev2={mu_prev2} - s0={s0} - norm_prev2={norm_prev2}")
                # print(f"mu_prev={mu_prev} - s1={s1} - norm_prev={norm_prev}")
                # print(f"mu_curr={mu_curr} - s2={s2} - norm_curr={norm_curr}")
                # print(f"munew={munew} - s_pred={snew} - norm_pred={norm_pred}")

                # now attempt to get a guesserror by an 'eval' run (cheap) in a temporary eval dir
                # guesserr = self.try_eval_guess(munew)
                # print(f"guesserr = {guesserr:.3g}")

                # break 
                # Decide whether to continue adjusting the guess, search on the guess, or give up.
                if prev_search_failed and ds < DS_MIN:
                    print("Stopping continuation because continuing would require ds <= DS_MIN")
                    break
                elif prev_search_failed and ds >= DS_MIN:
                    print("Stopping guess adjustments because previous search failed,")
                    print("so we're allowing guesserr <= GUESS_ERR_MIN, as long as DS_MIN <= ds")
                    break
                # Previous search succeeded.
                if GUESS_ERR_MIN <= guesserr <= GUESS_ERR_MAX:
                    print("Stopping guess adjustments because guess error is in bounds: GUESS_ERR_MIN <= guesserr <= GUESS_ERR_MAX")
                    break
                elif ds_reached_bounds:
                    print("Stopping guess adjustments because ds has reached its bounds:")
                    break
                elif no_more_adjustments:
                    print("Stopping guess adjustments because a recent search failed and we're being cautious.")
                    break
                elif guesserr_prev_search <= GUESS_ERR_MIN:
                    print("Previous search succeeded with guesserr <= GUESS_ERR_MIN.")
                    print("Let's try increasing ds, to lesser of 2*ds and value suggested by guesserr = O(ds^3)")
                    print("but no more than max allowed value DS_MAX. And no more guess adjustments after this.")
                    ds = min(2.0 * ds, ds * (GUESS_ERR_MIN / guesserr) ** (1/3))
                    if ds > DS_MAX:
                        ds = DS_MAX
                        ds_reached_bounds = True
                    else:
                        no_more_adjustments = True
                    continue
                else:
                    print("Guess not yet within bounds. Try new guess based on model guesserr = O(ds^3)")
                    print(f"guesserrtarget == {guesserrtarget}")
                    print(f"guesserr       == {guesserr}")
                    print(f"(err/targ)^1/3 == {(guesserrtarget / guesserr) ** (1/3)}")
                    ds *= (guesserrtarget / guesserr) ** (1/3)
                    if ds < DS_MIN:
                        ds = DS_MIN
                        ds_reached_bounds = True
                    elif ds > DS_MAX:
                        ds = DS_MAX
                        ds_reached_bounds = True
                    continue
            # End of guess adjustment loop
            # Done looking for a good guess. Now search on the guess.
            # break
            # run ECS with predicted mu and previous solution as restart
            xnew, resnew, obsnew = self.run_ecs(xnew)
            with open(self.odir+'mu.txt', 'w') as f:
                f.write(f"{self.print_mu()}\n")
                f.close()
            # tol_new, norm_new = 0, 0.735492785
            if resnew>1e-10 or resnew is None:
                # print(f"Step {self.isearch} failed at {self.mu_name} = {munew}")
                print("Failed to find new solution. Halving ds and trying again")
                print(f"{'failure':>8}{snew:>{W}.6g}{munew:>{W}.6g}   residual == {resnew}")
                ds *= 0.5
                prev_search_failed = True
                guesserr_prev_search = guesserr

                # Save failed search directory for inspection
                failuredir = os.path.join(self.folder, "failures")
                if not os.path.exists(failuredir):
                    os.mkdir(failuredir)
                srcdir = self.odir
                dstdir = os.path.join(failuredir, f"search-{self.isearch}")
                if os.path.exists(srcdir):
                    try:
                        # Move the failed search directory
                        os.rename(srcdir, dstdir)
                    except Exception as e:
                        print(f"Failed to move {srcdir} to {dstdir}: {e}")
            else:
                print(f"Step {self.isearch} succeeded at {self.mu_name} = {munew} with obs = {obsnew} and res = {resnew}")
                
                
                ds = np.hypot((obsnew - obs[2]) / obsnorm, (munew - mu[2]) / munorm)
                snew = s[2] + ds
                update_3list(x, xnew)
                update_3list(s, snew)
                update_3list(mu, munew)
                update_3list(obs, obsnew)
                update_3list(res, resnew)

                if self.ecs == "orb":
                    self.load_T()
                    Tnew = self.T
                    update_3list(T, Tnew)
                if self.xrel:
                    self.load_ax()
                    axnew = self.ax
                    update_3list(ax, axnew)
                if self.zrel:
                    self.load_az()
                    aznew = self.az
                    update_3list(az, aznew)

                property = self.load_flow_properties()
                with open(self.folder+'flow_properties.txt', 'a') as f:
                    f.write(f"{munew}, " + property + ", search-"+str(self.isearch)+"/" + "\n")
                    f.close()
                # update for next iteration
                self.inputpath = self.folder+"search-"+str(self.isearch)+"/solution.h5"
                prev_search_failed = False
                guesserr_prev_search = guesserr





# Initialize continuation object
continuation = ECSContinuation(ecs_name="eqb",
                               xrel=False,ax=0.5,
                               zrel=False,az=0.5,
                               mu_name="kx",
                               folder="SF2_continuation/",
                               guess="../Data/Checkpoint/EQ_SF2_kx8/solution.h5", 
                               kx=8, Lz=1, 
                               Ra=1e5, Pr=7, Rrho=40.0, tau=0.01,
                               T=0.01,dt=0.0001,adjust_dt=True,
                               eigen=False, 
                               symm=False,symmfile="",
                               n_procs=16)

# --- Normal continuation ---
# mu_list = np.linspace(14.0, 1, 66)
# continuation.natural_continuation(mu_list, restart_file=None)



# --- Pseudo-arclength continuation ---
continuation.arc_length_continuation(mu_start=8, dmu=0.1, n_steps=100, mu_target=20)