"""

To run the code using e.g. 16 processes:
    $ mpiexec -n 16 python3 find_ecs_v2.11_constraint.py --ecs=eqb --odir=debug_2.9/ --kx=14 --Lz=1 --T=1 --trust=1 --eigen=true --input=debug_eigen/solution.h5

"""

import h5py
import scipy.io
from scipy import sparse
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from dedalus import public as de

import os
import shutil
import glob
import sys


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import logging
logging.getLogger('solvers').setLevel(logging.WARNING)
logging.getLogger('subsystems').setLevel(logging.WARNING)
import timeit
start = timeit.default_timer()

#---------------------------------------- #
# Input parameters
#---------------------------------------- #
import argparse
parser = argparse.ArgumentParser(description="Find ECS with Dedalus")
parser.add_argument("--ecs", type=str, default=None, help="Type of ECS")
parser.add_argument("--lbvp", action='store_true', default=False, help="Enable LBVP solver for equilibria")
parser.add_argument("--xrel", action='store_true', default=False, help="Search for traveling wave or relative periodic orbit with shift in x")
parser.add_argument("--zrel", action='store_true', default=False, help="Search for traveling wave or relative periodic orbit with shift in z")
parser.add_argument("--odir", type=str, default="./", help="Output directory")
parser.add_argument("--kx", type=float, default=1.0, help="Horizontal wavenumber")
parser.add_argument("--kz", type=float, default=1.0, help="Vertical wavenumber")
parser.add_argument("--Lx", type=float, default=None, help="Domain size in x")
parser.add_argument("--Lz", type=float, default=None, help="Domain size in z")
parser.add_argument("--Ra", type=float, default=1e5, help="Rayleigh number")
parser.add_argument("--Pr", type=float, default=7.0, help="Prandtl number")
parser.add_argument("--Rrho", type=float, default=40.0, help="Density ratio T/S")
parser.add_argument("--Lambda", type=float, default=2.0, help="Density ratio S/T")
parser.add_argument("--tau", type=float, default=0.01, help="Diffusivity ratio")
parser.add_argument("--Ri", type=float, default=1.0, help="Richardson number")
parser.add_argument("--eigen", action='store_true', default=False, help="Compute linear stability")
parser.add_argument("--Ne", type=int, default=100, help="Arnoldi iterations")
parser.add_argument("--T", type=float, default=0.002, help="Time period")
parser.add_argument("--ax", type=float, default=0, help="Shift interval in x-direction")
parser.add_argument("--az", type=float, default=0, help="Shift interval in z-direction")
parser.add_argument("--Niter", type=int, default=50, help="Newton iterations")
parser.add_argument("--tol", type=float, default=1e-10, help="Convergence criterion")
parser.add_argument("--rtol", action='store_true', default=False, help="Relative convergence criterion |gx|/|x|")
parser.add_argument("--trust", type=float, default=20.0, help="Trust region radius")
parser.add_argument("--hookstep", action='store_true', default=False, help="Hookstep trust radius optimization")
parser.add_argument("--symm", action='store_true', default=False, help="Enable symmetry to impose")
parser.add_argument("--symmfile", type=str, default="", help="Symmetry file")
parser.add_argument("--adjust_dt", action='store_true', default=False, help="Adjust time step during flow map phi")
parser.add_argument("--dt", type=float, default=1e-4, help="Initial time step")
parser.add_argument("--dtmax", type=float, default=0.5, help="Maximal time step")
parser.add_argument("--dtmin", type=float, default=1e-7, help="Minimal time step")
parser.add_argument("--krylov_dim", type=int, default=50, help="(Max) Krylov subspace dimension")
parser.add_argument("--pCond", action='store_true', default=False, help="Enable phase condition")

parser.add_argument("--input", type=str, default="solution_guess.h5", help="Initial guess file")
args = parser.parse_args()

# set type of ecs
ecsmode = args.ecs
if ecsmode!="orb" and ecsmode!="eqb":
    ecsmode = None
lbvpmode = args.lbvp

# directory for saving data of ECS
ecs_dir = args.odir
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(ecs_dir):
        os.mkdir(ecs_dir)

# Now compute domain if not explicitly set
kx = args.kx
kz = args.kz
Lx = 2*np.pi/kx if args.Lx is None else args.Lx
Lz = 2*np.pi/kz if args.Lz is None else args.Lz
Ra, Pr, Rrho, Lambda, tau, Ri = args.Ra, args.Pr, args.Rrho, args.Lambda, args.tau, args.Ri
T_guess, ax_guess, az_guess = args.T, args.ax, args.az
newton_iterations = args.Niter
tolerance = args.tol
relative_tolerance = args.rtol
trust_radius = args.trust
hookstep = args.hookstep

adjust_dt = args.adjust_dt
max_dt = args.dtmax
min_dt = args.dtmin
dt_nominal = args.dt  # Nominal timestep. Actual timestep will differ due to rounding, so as to ensure an integer number of timesteps in the evaluation of the flow map phi
krylov_dim = args.krylov_dim # Max Krylov subspace dimension

Rxsearch = 1 if args.xrel is True else 0
Rzsearch = 1 if args.zrel is True else 0
Tsearch = 1 if ecsmode=="orb" else 0

ECS_eigen = args.eigen
Ne = args.Ne
symmetry = args.symm
symmetry_file = args.symmfile
if Rxsearch or Rzsearch:
    symmetry = False

PHASE_CONDITION = args.pCond
if MPI.COMM_WORLD.rank == 0:
    print("Phase condition:", PHASE_CONDITION)

symm=[1,1,1,1,0,0,0] # default: no symmetry
if symmetry:
    ''' Load symmetry from file '''
    ''' Example symmetry file format:
    1 1 1 1 0 0 0 : symmetry for s sx, sy, sz, ax, ay, az
    '''
    if symmetry_file!="":
        with open(symmetry_file, 'r') as f:
            symm_lines = f.readlines()
            symm[0] = float(symm_lines[0].split()[0])
            symm[1] = float(symm_lines[0].split()[1])
            symm[2] = float(symm_lines[0].split()[2])
            symm[3] = float(symm_lines[0].split()[3])
            symm[4] = float(symm_lines[0].split()[4])
            symm[5] = float(symm_lines[0].split()[5])
            symm[6] = float(symm_lines[0].split()[6])
    else:
        symm[4] = ax_guess
        symm[6] = az_guess

    if MPI.COMM_WORLD.rank == 0:
        print("Loaded symmetry:", symm)

def nontrivial_symm(symm):
    ''' Return true if this is nontrivial symmetry '''
    if symm[0]==1 and symm[1]==1 and symm[2]==1 and symm[3]==1 and symm[4]==0 and symm[5]==0 and symm[6]==0:
        return False
    else:
        return True

#---------------------------------------- #


from dedalus.tools.config import config
config['logging']['stdout_level'] = 'none'

#---------------------------------------- #
# Input file: initial guess for Newton solver
#---------------------------------------- #
input_filename = args.input
input_file = h5py.File(input_filename, 'r')
datat_init = np.array(input_file.get('/t')) # temperature
datas_init = np.array(input_file.get('/s')) # salinity
dataU_init = np.array(input_file.get('/u')) # x-velocity in grid space
dataW_init = np.array(input_file.get('/w')) # y-velocity in grid space
NX = np.shape(datat_init)[0] # resolution with dealias factor
NZ = np.shape(datat_init)[1] # resolution with dealias factor

#---------------------------------------- #
# Log file
#---------------------------------------- #
if MPI.COMM_WORLD.rank == 0:
    if os.path.exists(ecs_dir+'./solver.out'):
        os.remove(ecs_dir+'./solver.out')
    log_file = open(ecs_dir+'solver.out', 'w', buffering=1)

if MPI.COMM_WORLD.rank == 0:
    if os.path.exists(ecs_dir+'./flow_properties.txt'):
        os.remove(ecs_dir+'./flow_properties.txt')
#---------------------------------------- #

#---------------------------------------- #
# Solver parameters
#---------------------------------------- #
ECS_id = 'SF1' # (optional) identifier for ECS
output_full_trajectory = 1 # After the Newton solver converges, the code automatically time integrates over a single period and outputs
                                       # time series of some channel averages. There is also the option to output the full field data by setting 'output_full_trajectory' to 1

time_limit = 24
kmin = 40 # Minimum krylov subspace dimension
kfreq = 20 # Increment between 'kmin' and 'krylov_dim'

tr_min = 1e-8 # Minimum trust radius to try
timestepper = de.RK222

#---------------------------------------- #
# Bases and domain
#---------------------------------------- #
dealias_fac = 3/2 # Dealiasing 
coords = de.CartesianCoordinates('x','z')
dist = de.Distributor(coords, dtype=np.complex128)
x_basis = de.ComplexFourier(coords['x'], size=int(NX/dealias_fac), bounds=(0, Lx), dealias = dealias_fac)
z_basis = de.ComplexFourier(coords['z'], size=int(NZ/dealias_fac), bounds=(0, Lz), dealias = dealias_fac)
xg = x_basis.global_grid(dist, scale=dealias_fac)
zg = z_basis.global_grid(dist, scale=dealias_fac)

NXH = int(int(NX/dealias_fac)/2)
# Fields
p = dist.Field(name='p', bases=(x_basis,z_basis)) # create pressure field with scalar type
u = dist.VectorField(coords, name='u', bases=(x_basis,z_basis)) # create velocity field with vectoral type
sa = dist.Field(name='sa', bases=(x_basis,z_basis)) # create salinity field with scalar type
te = dist.Field(name='te', bases=(x_basis,z_basis)) # create temperature field with scalar type
tau_p = dist.Field(name='tau_p')  
tau_u = dist.VectorField(coords, name='tau_u') 
tau_te = dist.Field(name='tau_te') 
tau_sa = dist.Field(name='tau_sa') 

cx = dist.Field(name='cx')
cz = dist.Field(name='cz')

baru = dist.Field(bases=(z_basis)) # base flow
barT = dist.Field(bases=(z_basis)) # base state of temperature: y
barS = dist.Field(bases=(z_basis)) # base state of temperature: y


# Substitutions
x, z = dist.local_grids(x_basis, z_basis) # get coordinate arrays in horizontal and vertical directions
ex, ez = coords.unit_vector_fields(dist) # get unit vectors in horizontal and vertical directions
w = u @ ez # define velocity component in vertical direction

grad_te = de.grad(te) # First-order reduction
grad_sa = de.grad(sa) # First-order reduction
grad_u = de.grad(u) # First-order reduction

# First-order form: "lap(f)" becomes "div(grad_f)"
lap_u = de.div(grad_u)
lap_te = de.div(grad_te)
lap_sa = de.div(grad_sa)

dx = lambda A: de.Differentiate(A, coords['x']) 
dz = lambda A: de.Differentiate(A, coords['z']) 

lap = lambda A: de.div(de.grad(A))
grad = lambda A: de.grad(A)

h_mean = lambda A: de.Average(A,'x')
vol_avg = lambda A: de.Average(A)

baru['g'] = 0
barT['g'] = z
barS['g'] = z
totU = baru*ex + u
totT = barT + te
totS = barS + sa

### velocity nondimensionalization by thermal diffusivity
p1 = Pr
p2 = Pr*Ra
p3 = 1.0
p4 = 1.0/Rrho
p5 = 1.0
p6 = tau
# max_dt = 0.001
# time_sim_dt = 0.1
# snapshot_sim_dt = 0.1

# Nusselt and Sherwood numbers 
Nu = 1 - vol_avg(w*totT)/p5
Sh = 1 - vol_avg(w*totS)/p6
Ft = vol_avg(w*totT)/p5
Fs = vol_avg(w*totS)/p6
# Kinetic energy
KE = 0.5 * vol_avg(u@u)
# Dissipation rates


problem_phi = de.IVP([p, tau_p, u, te, sa, tau_u, tau_te, tau_sa], namespace= globals())
problem_phi.add_equation("trace(grad(u)) + tau_p = 0")
problem_phi.add_equation("integ(p) = 0") # Pressure gauge
problem_phi.add_equation("dt(u) + grad(p) - p1*lap_u - p2*(p3*te-p4*sa)*ez + tau_u = - u@grad_u")
problem_phi.add_equation("dt(te) - p5*lap_te + w + tau_te = - u@grad_te")
problem_phi.add_equation("dt(sa) - p6*lap_sa + w + tau_sa = - u@grad_sa")
problem_phi.add_equation("integ(u) = 0")
problem_phi.add_equation("integ(te) = 0")
problem_phi.add_equation("integ(sa) = 0")




##### for Linear boundary-value problem
u_rhs = dist.VectorField(coords, name='u_rhs', bases=(x_basis,z_basis))
sa_rhs = dist.Field(name='sa_rhs', bases=(x_basis,z_basis))
te_rhs = dist.Field(name='te_rhs', bases=(x_basis,z_basis))

# base state
u_eq = dist.VectorField(coords, name='u_eq', bases=(x_basis,z_basis))
sa_eq = dist.Field(name='sa_eq', bases=(x_basis,z_basis))
te_eq = dist.Field(name='te_eq', bases=(x_basis,z_basis))

# u_eq = u
# sa_eq = sa
# te_eq = te

# residual at base state
# u_rhs = - grad(p) + p1*lap(u_eq) + p2*(p3*te_eq-p4*sa_eq)*ez - u_eq@grad(u_eq)
# te_rhs = p5*lap(te_eq) - u_eq@ez - u_eq@grad(te_eq)
# sa_rhs = p6*lap(sa_eq) - u_eq@ez - u_eq@grad(sa_eq)

# build linearized equations
# Linearized LBVP (Jacobian · δx = -Residual)
problem_L = de.LBVP([p, tau_p, u, te, sa], namespace= globals())
problem_L.add_equation("trace(grad(u)) + tau_p = 0")
problem_L.add_equation("integ(p) = 0") # Pressure gauge
problem_L.add_equation("grad(p) - p1*lap_u - p2*(p3*te-p4*sa)*ez + u@grad(u_eq)+u_eq@grad(u)= - u_rhs")
problem_L.add_equation("- p5*lap_te + w + u@grad(te_eq)+u_eq@grad(te) = - te_rhs")
problem_L.add_equation("- p6*lap_sa + w + u@grad(sa_eq)+u_eq@grad(sa) = - sa_rhs")


# build nonlinear equations
# Nonlinear LBVP (Full residual F(x) = 0)
problem_NL = de.LBVP([p, tau_p, u, te, sa], namespace= globals())
problem_NL.add_equation("trace(grad(u)) + tau_p = 0")
problem_NL.add_equation("integ(p) = 0") # Pressure gauge
problem_NL.add_equation("grad(p) -p1*lap_u - p2*(p3*te-p4*sa)*ez = 0")
problem_NL.add_equation("- p5*lap_te + w = 0")
problem_NL.add_equation("- p6*lap_sa + w = 0")

#----------------------------------------------------------------------------------------- #
#----------------------------------------------------------------------------------------- #

#---------------------------------------- #
# Input field data for the initial guess
#---------------------------------------- #

u.load_from_global_grid_data(np.stack([dataU_init,dataW_init]))
te.load_from_global_grid_data(datat_init)
sa.load_from_global_grid_data(datas_init)

t_data_init = te.allgather_data('g').real
s_data_init = sa.allgather_data('g').real
U_data_init = u.allgather_data('g')[0].real
W_data_init = u.allgather_data('g')[1].real

#---------------------------------------- #
# Other parameters 
#---------------------------------------- #
n_fields = 4 # Number of physical fields (here: T, S, U, V)
MY = n_fields*NX*NZ # Total number of degrees of freedom
d_tol = 1e-7 # Step size for finite-difference in computing GMRES matrix-vector product
delta = 1e-7 # Step size for finite-difference approximation of the time derivative of the equations of motion
d_tol_translation = 1e-7 # Step size for finite-difference approximation of the infinitesimal generator of translations
n_timesteps = 10 # Number of timesteps used for the finite difference estimate  of the time derivative of the equations of motion
min_error = 1e-4

#---------------------------------------- #

def VectorToGrid(array_in):
    array_out = array_in.reshape(NX, NZ)
    return array_out
    
def GridToVector(array_in):
    array_out = array_in
    return array_out.ravel()



# 'f' is the state vector. Here we assign it the appropriate initial data. Note that we have to convert the Dedalus grid representation
# into a real-valued vector
Nunk = MY + Tsearch + Rxsearch + Rzsearch # of unknowns in x = [u,w,t,s] + constraints

# PHASE_CONDITION = False # enable phase condition
if PHASE_CONDITION:
    Nunk += 2  # add one more unknown for phase condition

# print('total Nunk = ',Nunk)
f = np.zeros(Nunk)
t_begin = 0
t_end = t_begin + NX*NZ
s_begin = t_end
s_end = s_begin + NX*NZ
u_begin = s_end
u_end = u_begin + NX*NZ
v_begin = u_end
v_end = v_begin + NX*NZ


f[t_begin:t_end] = t_data_init.ravel()
f[s_begin:s_end] = s_data_init.ravel()
f[u_begin:u_end] = U_data_init.ravel()
f[v_begin:v_end] = W_data_init.ravel()
if Tsearch:
    f[MY+Tsearch-1] = T_guess  # 'f[0]' gives the normalized period, i.e. the current value of 'T' divided by the initial guess, 'T_guess'
if Rxsearch:
    f[MY+Tsearch+Rxsearch-1] = ax_guess # similarly for the shift 'ax'
if Rzsearch:
    f[MY+Tsearch+Rxsearch+Rzsearch-1] = az_guess # similarly for the shift 'az'
if PHASE_CONDITION:
    f[-2] = 0.0  # initial guess for phase condition unknown
    f[-1] = 0.0

z0 = np.zeros(Nunk)
zn = np.copy(z0)
#---------------------------------------- #
# Load the field data from a state vector
#---------------------------------------- #
def load_state(array_in):
    te.load_from_global_grid_data(array_in[t_begin:t_end].reshape(NX, NZ))
    sa.load_from_global_grid_data(array_in[s_begin:s_end].reshape(NX, NZ))
    u.load_from_global_grid_data(np.stack([array_in[u_begin:u_end].reshape(NX, NZ),array_in[v_begin:v_end].reshape(NX, NZ)]))



def x_derivative(array_in):
    """Compute x-derivative of a field."""
    load_state(array_in)
    dxu = dx(u).evaluate().allgather_data('g').real
    dxt = dx(te).evaluate().allgather_data('g').real
    dxs = dx(sa).evaluate().allgather_data('g').real
    array_out = np.zeros(MY)
    array_out[u_begin:u_end] = dxu[0].ravel()
    array_out[v_begin:v_end] = dxu[1].ravel()
    array_out[t_begin:t_end] = dxt.ravel()
    array_out[s_begin:s_end] = dxs.ravel()
    return array_out

# set these once before the Newton loop starts
# if PHASE_CONDITION:
    # x_ref = np.copy(f[:MY]) # reference state vector for relative tolerance criterion
    # g_ref = x_derivative(x_ref) # reference derivative for relative tolerance criterion
    # gg = np.dot(g_ref, g_ref)
    # phase_scale = 1.0 / np.sqrt(gg + 1e-300)



def shift_x(field_u, ax, x_basis):
    """
    Shift fields by dx in the periodic x-direction.
    """
    kx = x_basis.wavenumbers
    phase_shift = np.exp(1j * kx * ax)
    field_u_coeff = field_u.allgather_data('c')
    field_u_coeff *= phase_shift[:, np.newaxis]
    field_u.load_from_global_coeff_data(field_u_coeff)

def shift_z(field_u, az, z_basis):
    """
    Shift fields by dz in the periodic z-direction.
    """
    kz = z_basis.wavenumbers
    phase_shift = np.exp(1j * kz * az)
    field_u_coeff = field_u.allgather_data('c')
    field_u_coeff *= phase_shift[np.newaxis, :]
    field_u.load_from_global_coeff_data(field_u_coeff)

def reflec_x(field_u, sign=1):
    """
    Reflec fields.
    """
    field_u_coeff = field_u.allgather_data('c')
    # code here
    field_u.load_from_global_coeff_data(field_u_coeff)


# Shift field data by 'd' units and convert to vector format
# def TransformG(array_in, d):
#     data_temp_c = np.copy(array_in)
#     d_grid = int( np.round(d / (Lx/int(NX*dealias_fac))) )
#     data_temp_c = np.roll(data_temp_c, shift=d_grid, axis=0)
#     return GridToVector(data_temp_c)

def TransformGx(array_in, ax):
    load_state(array_in)
    shift_x(u, ax=ax*Lx, x_basis=x_basis)
    shift_x(te, ax=ax*Lx, x_basis=x_basis)
    shift_x(sa, ax=ax*Lx, x_basis=x_basis)
    ug = u.allgather_data('g').real
    tg = te.allgather_data('g').real
    sg = sa.allgather_data('g').real
    data_temp_c = np.copy(array_in)
    data_temp_c[t_begin:t_end] = tg.ravel()
    data_temp_c[s_begin:s_end] = sg.ravel()
    data_temp_c[u_begin:u_end] = ug[0].ravel()
    data_temp_c[v_begin:v_end] = ug[1].ravel()
    return data_temp_c
def TransformGz(array_in, az):
    load_state(array_in)
    shift_z(u, az=az*Lz, z_basis=z_basis)
    shift_z(te, az=az*Lz, z_basis=z_basis)
    shift_z(sa, az=az*Lz, z_basis=z_basis)
    ug = u.allgather_data('g').real
    tg = te.allgather_data('g').real
    sg = sa.allgather_data('g').real
    data_temp_c = np.copy(array_in)
    data_temp_c[t_begin:t_end] = tg.ravel()
    data_temp_c[s_begin:s_end] = sg.ravel()
    data_temp_c[u_begin:u_end] = ug[0].ravel()
    data_temp_c[v_begin:v_end] = ug[1].ravel()
    return data_temp_c
# Compute infinitesimal generator of x-translation
def dxTransform(array_in):
    array_temp = np.copy(array_in)

    shifted_array_temp = TransformGx(array_temp, d_tol_translation)

    array_out = np.zeros(MY)
    array_out[t_begin:t_end] = (shifted_array_temp[t_begin:t_end] - array_temp[t_begin:t_end])/d_tol_translation
    array_out[s_begin:s_end] = (shifted_array_temp[s_begin:s_end] - array_temp[s_begin:s_end])/d_tol_translation
    array_out[u_begin:u_end] = (shifted_array_temp[u_begin:u_end] - array_temp[u_begin:u_end])/d_tol_translation
    array_out[v_begin:v_end] = (shifted_array_temp[v_begin:v_end] - array_temp[v_begin:v_end])/d_tol_translation
    return array_out
# Compute infinitesimal generator of z-translation
def dzTransform(array_in):
    array_temp = np.copy(array_in)

    shifted_array_temp = TransformGz(array_temp, d_tol_translation)

    array_out = np.zeros(MY)
    array_out[t_begin:t_end] = (shifted_array_temp[t_begin:t_end] - array_temp[t_begin:t_end])/d_tol_translation
    array_out[s_begin:s_end] = (shifted_array_temp[s_begin:s_end] - array_temp[s_begin:s_end])/d_tol_translation
    array_out[u_begin:u_end] = (shifted_array_temp[u_begin:u_end] - array_temp[u_begin:u_end])/d_tol_translation
    array_out[v_begin:v_end] = (shifted_array_temp[v_begin:v_end] - array_temp[v_begin:v_end])/d_tol_translation
    return array_out



''' Function phi:

    Description
    ---------------------
    Flow map phi(array_in, T, d)
    ---------------------

    Parameters
    ---------------------
    array_in :
        variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NZ
        description: state vector x0 in coefficient format
    T :
        variable type: float
        description: time interval over which to apply the flow map
    d :
        variable type: float
        description: shift to apply at the end of the time integration
    ---------------------
'''
def phi(array_in, T, ax, az):
    solver_phi = problem_phi.build_solver(timestepper)
    
    # copy the input data
    f = np.copy(array_in)
    u.load_from_global_grid_data(np.stack([f[u_begin:u_end].reshape(NX, NZ),f[v_begin:v_end].reshape(NX, NZ)]))
    te.load_from_global_grid_data(f[t_begin:t_end].reshape(NX, NZ))
    sa.load_from_global_grid_data(f[s_begin:s_end].reshape(NX, NZ))

    # set T
    solver_phi.stop_sim_time = T
    solver_phi.sim_time = 0
    solver_phi.iteration = 0
    solver_phi.stop_wall_time = np.inf
    solver_phi.stop_iteration = np.inf

    if adjust_dt:
        # set CFL
        CFL = de.CFL(solver_phi, initial_dt=dt_nominal, cadence=10, safety=0.5, threshold=0.05,
                    max_change=1.5, min_change=0.5, max_dt=max_dt, min_dt=min_dt)
        CFL.add_velocity(u)
        # run simulator
        while solver_phi.proceed:
            dt = CFL.compute_timestep()
            if solver_phi.sim_time + dt > T:
                dt = T - solver_phi.sim_time
            solver_phi.step(dt)
    else:
        # this way is better to run DNS over map of exact T
        # more stability
        num_steps = int(T/dt_nominal)
        dt = T/num_steps
        for i in range(num_steps):
            solver_phi.step(dt)

    ### future work, which include all symmetries (reflection and translation)
    if symmetry and nontrivial_symm(symm):
        if symm[4] != 0: # ax
            shift_x(u, ax=symm[4]*Lx, x_basis=x_basis)
            shift_x(te, ax=symm[4]*Lx, x_basis=x_basis)
            shift_x(sa, ax=symm[4]*Lx, x_basis=x_basis)
        if symm[6] != 0: # az 
            shift_z(u, az=symm[6]*Lz, z_basis=z_basis)
            shift_z(te, az=symm[6]*Lz, z_basis=z_basis)
            shift_z(sa, az=symm[6]*Lz, z_basis=z_basis)
    if Rxsearch:
        shift_x(u, ax=ax*Lx, x_basis=x_basis)
        shift_x(te, ax=ax*Lx, x_basis=x_basis)
        shift_x(sa, ax=ax*Lx, x_basis=x_basis)
    
    if Rzsearch:
        shift_z(u, az=az*Lz, z_basis=z_basis)
        shift_z(te, az=az*Lz, z_basis=z_basis)
        shift_z(sa, az=az*Lz, z_basis=z_basis)

    # gather data
    tg = te.allgather_data('g').real
    sg = sa.allgather_data('g').real
    ug = u.allgather_data('g').real

    # save data in an array
    array_out = np.zeros(MY)
    array_out[u_begin:u_end] = ug[0].ravel()
    array_out[v_begin:v_end] = ug[1].ravel()
    array_out[t_begin:t_end] = tg.ravel()
    array_out[s_begin:s_end] = sg.ravel()
    

    return array_out
''' '''

# Output data over a single period of the ECS
def phi_out(array_in, T, ax, az):
    solver_phi = problem_phi.build_solver(timestepper)
    
    # copy the input data
    f = np.copy(array_in)
    u.load_from_global_grid_data(np.stack([f[u_begin:u_end].reshape(NX, NZ),f[v_begin:v_end].reshape(NX, NZ)]))
    te.load_from_global_grid_data(f[t_begin:t_end].reshape(NX, NZ))
    sa.load_from_global_grid_data(f[s_begin:s_end].reshape(NX, NZ))

    # set T
    sim_time = T # for periodic orbit
    n_full_solution_steps = 100
    # for traveling wave and relative periodic orbit
    if Rxsearch and not Rzsearch:
        if abs(ax) < 1e-5:
            return
        sim_time = 1/abs(ax) * T if abs(ax)>0.2 else 2 * T
        n_full_solution_steps = 2*100
    elif not Rxsearch and Rzsearch:
        if abs(az) < 1e-5:
            return
        sim_time = 1/abs(az) * T if abs(az)>0.2 else 2 * T
        n_full_solution_steps = 2*100
    elif Rxsearch and Rzsearch:
        if abs(ax) < 1e-5 and abs(az)<1e-5:
            return
        if abs(ax)<0.2 and abs(az)<0.2:
            sim_time = 2 * T
        else:
            sim_time = 1/abs(ax) * T if 1/abs(ax) < 1/abs(az) else 1/abs(az) * T
        n_full_solution_steps = 2*100

    solver_phi.stop_sim_time = sim_time
    solver_phi.sim_time = 0
    solver_phi.iteration = 0
    solver_phi.stop_wall_time = np.inf
    solver_phi.stop_iteration = np.inf

    # n_full_solution_steps = 2*100 if (Rxsearch or Rzsearch) else 100
    
    
    if output_full_trajectory == 1:
        full_solution = solver_phi.evaluator.add_file_handler(ecs_dir+'full_solution', sim_dt=sim_time/n_full_solution_steps)
        full_solution.add_task(te, layout='g', name='t')
        full_solution.add_task(sa, layout='g', name='s')
        full_solution.add_task(u@ex, layout='g', name='u')
        full_solution.add_task(u@ez, layout='g', name='w')
        full_solution.add_task(Nu, name='Nu')
        full_solution.add_task(Sh, name='Sh')

        # order_params = solver_phi.evaluator.add_file_handler(ecs_dir+'order_params', sim_dt=T/n_full_solution_steps, max_writes=np.inf)
        # order_params.add_task(Nu, name='Nu') # get Nu
        # order_params.add_task(Sh, name='Sh') # get Sh


    if adjust_dt:
        # set CFL
        CFL = de.CFL(solver_phi, initial_dt=dt_nominal, cadence=10, safety=0.5, threshold=0.05,
                    max_change=1.5, min_change=0.5, max_dt=max_dt, min_dt=min_dt)
        CFL.add_velocity(u)
        # run simulator
        while solver_phi.proceed:
            dt = CFL.compute_timestep()
            if solver_phi.sim_time + dt > sim_time:
                dt = sim_time - solver_phi.sim_time
            solver_phi.step(dt)
    else:
        # this way is better to run DNS over map of exact T
        # more stability
        num_steps = int(sim_time/dt_nominal)
        dt = sim_time/num_steps
        for i in range(num_steps):
            solver_phi.step(dt)




''' Function Dphi_prod:

    Description
    ---------------------
    Matrix-vector product, math notation: (d phi / dx_0) * delta_x 
    ---------------------

    Parameters
    ---------------------
    array_base :
        variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NZ
        description: initial (base) vector x0 in coefficient format
    array_pert :
        variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NZ
        description: 'delta_x' in the equation for the Matrix-vector product
    T :
        variable type: float
        description: time interval over which to apply the flow map
    d :
        variable type: float
        description: shift to apply at the end of the time integration
    ---------------------
'''
def Dphi_prod(array_base, array_pert, phi_base, T0, ax0, az0):
    norm_v = np.linalg.norm(array_pert)
    if norm_v == 0:
        return np.zeros_like(array_pert)
    epsilon = d_tol / norm_v

    array_init = array_base + epsilon*array_pert
    array_final = phi(array_init, T0, ax0, az0)

    array_out = (array_final-phi_base)/epsilon
    
    # IMPORTANT: remove symmetry (neutral) component
    # g = x_derivative(array_base)
    # array_out = project_out(array_out, g)
    return array_out
''' '''



''' Function applyLinearOperator:

    Description
    ---------------------
    Linearized operator for the Newton iteration
    ---------------------

    Parameters
    ---------------------
    array_base :
        variable type: Real-valued, 1d numpy data array with dimension MY + 2 = n_fields*NX*NZ + 2
        description: base vector about which the linearization is performed, including the period 'T' and shift 'd'
    array_pert :
        variable type: Real-valued, 1d numpy data array with dimension MY + 2 = n_fields*NX*NZ + 2
        description: linear perturbation we are solving for, i.e.,  'dx' in the equation 'A*dx = b'
    ---------------------
'''
def applyLinearOperator(array_base, array_pert, phi_base):
    if lbvpmode==True and ecsmode == "eqb" and Rxsearch==False and Rzsearch==False:
        # use LBVP for equilibrium
        solverL = problem_L.build_solver()

        x_base = np.copy(array_base)
        delta_x = np.copy(array_pert)
        
        # copy the base state
        u_eq.load_from_global_grid_data(np.stack([x_base[u_begin:u_end].reshape(NX, NZ),
                                        x_base[v_begin:v_end].reshape(NX, NZ)]))
        te_eq.load_from_global_grid_data(x_base[t_begin:t_end].reshape(NX,NZ))
        sa_eq.load_from_global_grid_data(x_base[s_begin:s_end].reshape(NX,NZ))

        # copy the perturbation state
        u.load_from_global_grid_data(np.stack([delta_x[u_begin:u_end].reshape(NX, NZ),
                                    delta_x[v_begin:v_end].reshape(NX, NZ)]))
        te.load_from_global_grid_data(delta_x[t_begin:t_end].reshape(NX,NZ))
        sa.load_from_global_grid_data(delta_x[s_begin:s_end].reshape(NX,NZ))

        # set the rhs
        u_rhs['g'] = - grad(p) + p1*lap(u_eq) + p2*(p3*te_eq-p4*sa_eq)*ez - u_eq@grad(u_eq)
        te_rhs['g'] = p5*lap(te_eq) - u_eq@ez - u_eq@grad(te_eq)
        sa_rhs['g'] = p6*lap(sa_eq) - u_eq@ez - u_eq@grad(sa_eq)

        solverL.solve()

        array_out = np.zeros(MY)
        array_out[u_begin:u_end] = u.allgather_data('g')[0].real.ravel()
        array_out[v_begin:v_end] = u.allgather_data('g')[1].real.ravel()
        array_out[t_begin:t_end] = te.allgather_data('g').real.ravel()
        array_out[s_begin:s_end] = sa.allgather_data('g').real.ravel()
        
        return array_out
    else:
        T0 = T_guess
        ax0 = ax_guess
        az0 = az_guess
        if Tsearch:
            T0 = array_base[MY+Tsearch-1]
            delta_T = array_pert[MY+Tsearch-1]
        if Rxsearch:
            ax0 = array_base[MY+Tsearch+Rxsearch-1]
            delta_dx = array_pert[MY+Tsearch+Rxsearch-1]
        if Rzsearch:
            az0 = array_base[MY+Tsearch+Rxsearch+Rzsearch-1]
            delta_dz = array_pert[MY+Tsearch+Rxsearch+Rzsearch-1]
        
        x_base = np.copy(array_base[:MY])
        delta_x = np.copy(array_pert[:MY])
        
        array_out = np.zeros(Nunk)
        array_out[:MY] = Dphi_prod(x_base, delta_x, phi_base, T0, ax0, az0) - delta_x 
        if Tsearch:
            array_out[:MY] += RHS(np.copy(phi_base))*delta_T 
            array_out[MY+Tsearch-1] = np.matmul(np.conj(RHS(x_base)), delta_x)
        if Rxsearch:
            array_out[:MY] += dxTransform(phi_base)*delta_dx
            array_out[MY+Tsearch+Rxsearch-1] = np.matmul(np.conj(dxTransform(x_base)), delta_x)
        if Rzsearch:
            array_out[:MY] += dzTransform(phi_base)*delta_dz
            array_out[MY+Tsearch+Rxsearch+Rzsearch-1] = np.matmul(np.conj(dzTransform(x_base)), delta_x)
        # ===== PHASE CONDITION JACOBIAN =====
        # if PHASE_CONDITION:
        #     delta_alpha = array_pert[-1]
        #     array_out[:MY] += g_ref * delta_alpha # Column for alpha: ∂(Ru)/∂alpha = g_ref
        #     array_out[-1] = phase_scale * np.dot(delta_x, g_ref) # Row for phase condition: ∂g/∂x · δx = g_ref · δx
        return array_out
''' '''



''' Function RHS:

    Description
    ---------------------
    Calculate time-derivative in the final state via finite difference (equal to the r.h.s. of the equations of motion)
    ---------------------

    Parameters
    ---------------------
    array_in :
        variable type: Real-valued, 1d numpy data array with dimension MY = n_fields*NX*NZ
        description: input vector
    ---------------------
'''
def RHS(array_in):
    
    solver_phi = problem_phi.build_solver(de.RK222)
    
    # copy the input data
    f = np.copy(array_in)
    u.load_from_global_grid_data(np.stack([f[u_begin:u_end].reshape(NX, NZ),f[v_begin:v_end].reshape(NX, NZ)]))
    te.load_from_global_grid_data(f[t_begin:t_end].reshape(NX, NZ))
    sa.load_from_global_grid_data(f[s_begin:s_end].reshape(NX, NZ))

    t_data_initial = te.allgather_data('g').real
    s_data_initial = sa.allgather_data('g').real
    U_data_initial = u.allgather_data('g')[0].real
    V_data_initial = u.allgather_data('g')[1].real

    for i in range(n_timesteps):
        solver_phi.step(delta)

    tg = te.allgather_data('g').real
    sg = sa.allgather_data('g').real
    ug = u.allgather_data('g').real

    array_out = np.zeros(MY)
    array_out[u_begin:u_end] = GridToVector((np.copy(ug[0]) - U_data_initial)/(n_timesteps*delta))
    array_out[v_begin:v_end] = GridToVector((np.copy(ug[1]) - V_data_initial)/(n_timesteps*delta))
    array_out[t_begin:t_end] = GridToVector((np.copy(tg) - t_data_initial)/(n_timesteps*delta))
    array_out[s_begin:s_end] = GridToVector((np.copy(sg) - s_data_initial)/(n_timesteps*delta))
    return array_out
''' '''




''' Function applyNonLinearOperator:

    Description
    ---------------------
    Full nonlinear operator
    ---------------------

    Parameters
    ---------------------
    array_in :
        variable type: Real-valued, 1d numpy data array with dimension MY + 2 = n_fields*NX*NZ + 2
        description: input vector
    ---------------------
'''
def applyNonLinearOperator(array_in):
    # if ecsmode=="eqb" and Rxsearch==False and Rzsearch==False:
    #     # use LBVP for equilibrium
    #     solverNL = problem_NL.build_solver()

    #     f = np.copy(array_in)
        
    #     # copy the perturbation state
    #     u.load_from_global_grid_data(
    #         np.stack([f[u_begin:u_end].reshape(NX, NZ),
    #                 f[v_begin:v_end].reshape(NX, NZ)])
    #     )
    #     te.load_from_global_grid_data(f[t_begin:t_end].reshape(NX,NZ))
    #     sa.load_from_global_grid_data(f[s_begin:s_end].reshape(NX,NZ))

    #     solverNL.solve()

    #     # gather data
    #     array_out = np.zeros(MY)
    #     array_out[u_begin:u_end] = u.allgather_data('g')[0].real.ravel()
    #     array_out[v_begin:v_end] = u.allgather_data('g')[1].real.ravel()
    #     array_out[t_begin:t_end] = te.allgather_data('g').real.ravel()
    #     array_out[s_begin:s_end] = sa.allgather_data('g').real.ravel()

    #     return array_out
    # else:
    array_out = np.zeros(Nunk)
    if Tsearch:
        T_temp = array_in[MY+Tsearch-1]
    else:
        T_temp = T_guess

    if Rxsearch:
        ax_temp = array_in[MY+Tsearch+Rxsearch-1]
    else:
        ax_temp = ax_guess

    if Rzsearch:
        az_temp = array_in[MY+Tsearch+Rxsearch+Rzsearch-1]
    else:
        az_temp = az_guess

    array_out[:MY] = -phi(array_in[:MY], T_temp, ax_temp, az_temp) + array_in[:MY]

    # ===== PHASE CONDITION =====
    # if PHASE_CONDITION:
    #     x = array_in[:MY]
    #     alpha = array_in[-1]
    #     array_out[:MY] += alpha * g_ref
    #     array_out[-1] = phase_scale * np.dot(x - x_ref, g_ref)

    return array_out
''' '''
# def arnoldi_iteration(x_base, phi_base, T, d, r, n):
#     Q = np.zeros((r.size, n+1), dtype=complex)
#     H = np.zeros((n+1, n), dtype=complex)

#     Q[:, 0] = r / np.linalg.norm(r)

#     for k in range(1, n+1):

#         # Krylov vector
#         Q[:, k] = Dphi_prod(x_base, Q[:, k-1], phi_base, T, d)

#         # Modified Gram–Schmidt
#         for j in range(k):
#             H[j, k-1] = np.vdot(Q[:, j], Q[:, k])
#             Q[:, k] -= H[j, k-1] * Q[:, j]

#         # Reorthogonalize (important!)
#         for j in range(k):
#             h2 = np.vdot(Q[:, j], Q[:, k])
#             H[j, k-1] += h2
#             Q[:, k] -= h2 * Q[:, j]

#         # Normalize
#         H[k, k-1] = np.linalg.norm(Q[:, k])
#         if H[k, k-1] < 1e-14:
#             print("Arnoldi breakdown at k =", k)
#             return Q[:, :k], H[:k+1, :k]

#         Q[:, k] /= H[k, k-1]

#     return Q, H

def project_out(v, g):
    """Remove component of v along g: v <- v - g*(g·v)/(g·g)."""
    gg = np.dot(g, g)
    if gg == 0:
        return v
    return v - g * (np.dot(g, v) / gg)
def arnoldi_iteration(x_base, phi_base, T, ax, az, r, n:int):
    # g = x_derivative(x_base)          # group tangent (neutral direction)
    # r = project_out(r, g)             # ensure start vector not aligned with neutral dir

    Q = np.zeros((r.size, n+1))
    H = np.zeros((n+1, n))
    Q[:,0] = r/np.linalg.norm(r)

    for k in range(1, n + 1):
    #     Q[:,k] = Dphi_prod(x_base, Q[:, k - 1], phi_base, T, d)
    #     for j in range(0, k):
    #         H[j, k-1] = np.matmul(np.conj(Q[:,j]), Q[:,k])
    #         Q[:,k] = Q[:,k] - H[j, k-1]*Q[:,j]
    #     H[k, k-1] = np.linalg.norm(Q[:,k])
    #     Q[:,k] = Q[:,k]/H[k, k-1]

        v = Dphi_prod(x_base, Q[:, k - 1], phi_base, T, ax, az)
        # v = project_out(v, g)         # <- key: remove neutral component here
        for j in range(0, k):
            H[j, k-1] = np.vdot(Q[:,j], v)
            v = v - H[j, k-1]*Q[:,j]
        H[k, k-1] = np.linalg.norm(v)
        Q[:,k] = v/H[k, k-1]
        
    return Q, H
    
def arnoldi_iteration_inner(x_base, Q, phi_base, k:int):
    Qk = applyLinearOperator(x_base, Q[:, k - 1], phi_base)
    Hk = np.zeros(k+1)
    for j in range(0, k):
        Hk[j] = np.matmul(np.conj(Q[:,j]), Qk)
        Qk = Qk- Hk[j]*Q[:,j]
    Hk[k] = np.linalg.norm(Qk)
    Qk = Qk/Hk[k]

    return Qk, Hk
    
def Hookstep(H_, beta_, k_, tr):
    e1 = np.zeros(k_+1)
    e1[0] = beta_
    
    def fun(x_, F):
        r = np.matmul(F, x_) + e1
        return np.matmul(r, r)

    def Jacobian(x_, F):
        return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
    
    def constraint(x_):
        return tr*tr - np.matmul(np.transpose(x_), x_)
    
    def constraintJac(x_):
        return -2*x_
    
    ineq_cons = {'type': 'ineq','fun' : constraint,'jac' : constraintJac}
    
    w_init = np.zeros(k_)
    w_init[0] = 1e-3
    
    res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
        constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 100000000}, bounds=None)
    return res

# Used for the Hookstep; will be documented in detail in a later release
def Gmin(xb_, x0_, k_, beta_, tr_, Q_, H_):
	e1 = np.zeros(k_+1)
	e1[0] = beta_
	
	def fun(x_, F):
		r = np.matmul(F, x_) + e1
		return np.matmul(r, r)

	def Jacobian(x_, F):
		return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
	
	def constraint(x_):
		return tr_*tr_ - np.matmul(np.transpose(x_), x_)
	
	def constraintJac(x_):
		return -2*x_
	
	ineq_cons = {'type': 'ineq', 'fun' : constraint, 'jac' : constraintJac}
	
	w_init = np.zeros(k_)
	w_init[0] = 1e-3
	
	res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
							constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 10000}, bounds=None)
	xk = np.matmul(Q_[:,0:k_], res.x)
	return np.linalg.norm(applyNonLinearOperator(np.copy(xb_)+np.copy(x0_)+xk))

# Used for the Hookstep; will be documented in detail in a later release
def Gmin_full(xb_, x0_, k_, beta_, tr_, Q_, H_):
	e1 = np.zeros(k_+1)
	e1[0] = beta_
	
	def fun(x_, F):
		r = np.matmul(F, x_) + e1
		return np.matmul(r, r)

	def Jacobian(x_, F):
		return 2*np.matmul(np.matmul(np.transpose(F), F), x_) + 2*np.matmul(np.transpose(F), e1)
	
	def constraint(x_):
		return tr_*tr_ - np.matmul(np.transpose(x_), x_)
	
	def constraintJac(x_):
		return -2*x_
	
	ineq_cons = {'type': 'ineq', 'fun' : constraint, 'jac' : constraintJac}
	
	w_init = np.zeros(k_)
	w_init[0] = 1e-3
	
	res = scipy.optimize.minimize(fun, w_init, args=(H_[0:k_+1,0:k_]), method='SLSQP', jac = Jacobian,
							constraints=(ineq_cons), options={'ftol': 1e-34, 'disp': False, 'maxiter': 10000}, bounds=None)
	xk = np.matmul(Q_[:,0:k_], res.x)
	return xk, np.linalg.norm(applyNonLinearOperator(np.copy(xb_)+np.copy(x0_)+xk))

# Used for the Hookstep; will be documented in detail in a later release
def TRmin(xb, x0_, k_, beta_, tr0, Q_, H_):
	tr = np.copy(tr0)
	counter = 0
	while tr > tr_min:
		tr = 0.5*tr
		counter = counter + 1
	x_data = np.zeros(counter)
	y_data = np.zeros(counter)
	tr = np.copy(tr0)
	counter = 0
	while tr > tr_min:
		x_data[counter] = tr
		y_data[counter] = Gmin(xb, x0_, k_, beta_, tr, Q_, H_)
		counter = counter + 1
		tr = 0.5*tr
	return x_data[np.argmin(y_data)], np.min(y_data)

def GMRES(x_base, x0, phi_base, b, kmax, tr):
    if lbvpmode==True and ecsmode=="eqb" and Rxsearch==False and Rzsearch==False:
        x0_cached = np.copy(x0)
        r = applyLinearOperator(x_base, x0, phi_base) - b
        rho = np.linalg.norm(r)
        beta = rho
        b_norm = np.linalg.norm(b)

        Q = np.zeros((x0.size, kmax+1))
        H = np.zeros((kmax+1, kmax))

        min_vector = np.copy(x0)
        MINR = np.inf
        MIN_TR = np.inf
        Q[:,0] = r/np.linalg.norm(r)
        for k in range(1, kmax+1):
            Q[:,k], H[:k+1,k-1] = arnoldi_iteration_inner(x_base, Q[:,0:k], k)

            if k > kmin and k % kfreq == 0:
                tr_min, min_res = TRmin(np.copy(x_base), x0, k, beta, trust_radius, Q, H)
                xk, error = Gmin_full(np.copy(x_base), x0, k, beta, tr_min, Q, H)
                if error < MINR:
                    MINR = error
                    min_vector = np.copy(xk)
                    MIN_TR = tr_min

        return x0_cached + min_vector, MINR, MIN_TR
    else:
        xk = np.copy(x0)
        r = applyLinearOperator(x_base, x0, phi_base) - b
        rho = np.linalg.norm(r)
        beta = rho
        b_norm = np.linalg.norm(b)
        
        Q = np.zeros((x0.size, kmax+1))
        H = np.zeros((kmax+1, kmax))
        
        min_error = np.inf
        min_vector = np.zeros(x0.size)
        
        Q[:,0] = r/np.linalg.norm(r)
        for k in range(1, kmax):
            Q[:,k], H[:k+1,k-1] = arnoldi_iteration_inner(x_base, Q[:,0:k], phi_base, k)
            res = Hookstep(H, beta, k, tr)
            rho = np.linalg.norm(res.fun)
            if MPI.COMM_WORLD.rank == 0:
                print(".", end='', flush=True)
            if rho < min_error:
                min_error = rho
                xk = np.matmul(Q[:,0:k], res.x)

                
        test = np.linalg.norm(applyNonLinearOperator(np.copy(x_base)+x0+xk))
        tr_local = tr
        while test > 0.99*b_norm and tr_local > 1e-10:
            res = Hookstep(H, beta, kmax - 1, tr_local)
            xk = np.matmul(Q[:,0:(kmax-1)], res.x)
            min_error = np.linalg.norm(res.fun)
            test = np.linalg.norm(applyNonLinearOperator(np.copy(x_base)+x0+xk))
            tr_local = 0.5*tr_local
        return x0 + xk, min_error, tr_local

def save_flow_properties(f):
    # copy the input data
    u.load_from_global_grid_data(
        np.stack([f[u_begin:u_end].reshape(NX, NZ),
                f[v_begin:v_end].reshape(NX, NZ)])
    )
    te.load_from_global_grid_data(f[t_begin:t_end].reshape(NX, NZ))
    sa.load_from_global_grid_data(f[s_begin:s_end].reshape(NX, NZ))
    T_grid_data_out = te.allgather_data('g').real
    S_grid_data_out = sa.allgather_data('g').real
    U_grid_data_out = u.allgather_data('g')[0].real
    W_grid_data_out = u.allgather_data('g')[1].real

    L2u = np.linalg.norm([U_grid_data_out,W_grid_data_out])
    L2t = np.linalg.norm(T_grid_data_out)
    L2s = np.linalg.norm(S_grid_data_out)

    # Nusselt and Sherwood numbers 
    Nu_val = np.squeeze(Nu.evaluate()['g']).real
    Sh_val = np.squeeze(Sh.evaluate()['g']).real
    # Kinetic energy
    KE_val = np.squeeze(KE.evaluate()['g']).real

    # save to file
    if dist.comm.rank == 0:
        # If file does not exist, create and write header
        if not os.path.exists(ecs_dir+'flow_properties.txt'):
            with open(ecs_dir+'flow_properties.txt', 'w') as file:
                if Tsearch:
                    file.write("T, ")
                if Rxsearch:
                    file.write("ax, ")
                if Rzsearch:
                    file.write("az, ")
                file.write("L2u, L2t, L2s, KE, Nu, Sh\n")

        flow_file = open(ecs_dir+'flow_properties.txt', 'a')
        if Tsearch:
            flow_file.write(f"{f[MY+Tsearch-1]}, ")
        if Rxsearch:
            flow_file.write(f"{f[MY+Tsearch+Rxsearch-1]}, ")
        if Rzsearch:
            flow_file.write(f"{f[MY+Tsearch+Rxsearch+Rzsearch-1]}, ")
        flow_file.write(f"{L2u}, {L2t}, {L2s}, {KE_val}, {Nu_val}, {Sh_val}\n")
        flow_file.close()
    # save tolerance
    norm_file = open(ecs_dir+'norm.txt', 'w')
    norm_file.writelines(str(L2u))
    norm_file.close()
   
#---------------------------------------- #
# Save the field data and parameter to a file .h5
#---------------------------------------- #
def save_solution(f, final_error=0, name = 'solution'):
    # copy the input data
    u.load_from_global_grid_data(np.stack([f[u_begin:u_end].reshape(NX, NZ),f[v_begin:v_end].reshape(NX, NZ)]))
    te.load_from_global_grid_data(f[t_begin:t_end].reshape(NX, NZ))
    sa.load_from_global_grid_data(f[s_begin:s_end].reshape(NX, NZ))
    T_grid_data_out = te.allgather_data('g').real
    S_grid_data_out = sa.allgather_data('g').real
    U_grid_data_out = u.allgather_data('g')[0].real
    W_grid_data_out = u.allgather_data('g')[1].real
    xg = x_basis.global_grid(dist, scale=dealias_fac) 
    zg = z_basis.global_grid(dist, scale=dealias_fac)

    ### save state to .h5 file
    if dist.comm.rank == 0:
        h5f = h5py.File(ecs_dir+name+'.h5', 'w')
        #------------ Simulation parameters -------------------- #
        h5f.create_dataset('/ECS_id', data = ECS_id)
        h5f.create_dataset('/params/Ra', data = Ra)
        h5f.create_dataset('/params/Pr', data = Pr)
        h5f.create_dataset('/params/Rrho', data = Rrho)
        h5f.create_dataset('/params/Lambda', data = Lambda)
        h5f.create_dataset('/params/tau', data = tau)
        h5f.create_dataset('/params/Ri', data = Ri)
        h5f.create_dataset('/params/Lx', data = Lx)
        h5f.create_dataset('/params/Lz', data = Lz)
        h5f.create_dataset('/params/Nx', data = NX)
        h5f.create_dataset('/params/Nz', data = NZ)
        h5f.create_dataset('/params/dt', data = dt_nominal)
        h5f.create_dataset('/params/krylov_dim', data = krylov_dim)
        if Tsearch:
            h5f.create_dataset('T', data = f[MY+Tsearch-1])
        if Rxsearch:
            h5f.create_dataset('ax', data = f[MY+Tsearch+Rxsearch-1])
        if Rzsearch:
            h5f.create_dataset('az', data = f[MY+Tsearch+Rxsearch+Rzsearch-1])
        h5f.create_dataset('res', data = final_error)
        #------------ Grid space data -------------------- #
        h5f.create_dataset('xg', data = xg)
        h5f.create_dataset('zg', data = zg)
        h5f.create_dataset('t', data = T_grid_data_out)
        h5f.create_dataset('s', data = S_grid_data_out)
        h5f.create_dataset('u', data = U_grid_data_out)
        h5f.create_dataset('w', data = W_grid_data_out)
        #--------------------------------------------------------------------- #
        h5f.close()

    ### save properties of state

        

def plot_solution(f, name = "solution"):
    load_state(f[:MY])
    
    xg = x_basis.global_grid(dist, scale=dealias_fac)
    zg = z_basis.global_grid(dist, scale=dealias_fac)
    ug = u.allgather_data('g').real
    tg = te.allgather_data('g').real
    sg = sa.allgather_data('g').real
    if dist.comm.rank == 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        fig.suptitle(name)
        pcm1 = ax1.pcolormesh(xg.ravel(),zg.ravel(), ug[0].T, cmap='bwr', shading='auto')
        ax1.set_title("u")
        fig.colorbar(pcm1, ax=ax1, label=None)
        pcm2 = ax2.pcolormesh(xg.ravel(),zg.ravel(), ug[1].T, cmap='bwr', shading='auto')
        ax2.set_title("w")
        fig.colorbar(pcm2, ax=ax2, label=None)
        pcm3 = ax3.pcolormesh(xg.ravel(),zg.ravel(), tg.T, cmap='bwr', shading='auto')
        ax3.set_title("t")
        fig.colorbar(pcm3, ax=ax3, label=None)
        pcm4 = ax4.pcolormesh(xg.ravel(),zg.ravel(), sg.T, cmap='bwr', shading='auto')
        ax4.set_title("s")
        fig.colorbar(pcm4, ax=ax4, label=None)
        for ax in [ax1, ax2, ax3, ax4]:   # only data subplots
            ax.set(xlabel="x", ylabel="z")
        for ax in fig.get_axes():
            ax.label_outer()
        plt.savefig(ecs_dir+name+".png", dpi=300)
        plt.close()



#--------------------------------------------------------------------- #
# Check for symmetries by computing the residual ||S*f - f||, 
# where S is the symmetry operator and f is the state.
#--------------------------------------------------------------------- #
def TranslateTest(fields_data, n):
    shift = Lx/n
    fields_data_t = [np.copy(fields_data[0]), np.copy(fields_data[1]), np.copy(fields_data[2]), np.copy(fields_data[3])]
    for k in range(4):
        for i in range(NXH):
            fields_data_t[k][i,:] = np.exp(2*1j*i*np.pi*(shift/Lx))*fields_data_t[k][i,:]
    D = 0
    for k in range(4):
        D = D + np.linalg.norm(fields_data_t[k] - fields_data[k])
    return D
    
def S1TranslateTest(fields_data, n):
    shift = Lx/n
    fields_data_t = [np.copy(fields_data[0]), np.copy(fields_data[1]), np.copy(fields_data[2]), np.copy(fields_data[3])]
    
    for k in [0,2]:
        for j in range(NZ):
            if j % 2 == 0:
                fields_data_t[k][:,j] = fields_data_t[k][:,j]
            elif j % 2 == 1:
                fields_data_t[k][:,j] = -fields_data_t[k][:,j]
                
    for k in [1,3]:
        for j in range(NZ):
            if j % 2 == 0:
                fields_data_t[k][:,j] = -fields_data_t[k][:,j]
            elif j % 2 == 1:
                fields_data_t[k][:,j] = fields_data_t[k][:,j]
    
    for k in range(4):
        for i in range(NXH):
            fields_data_t[k][i,:] = np.exp(2*1j*i*np.pi*(shift/Lx))*fields_data_t[k][i,:]
    D = 0
    for k in range(4):
        D = D + np.linalg.norm(fields_data_t[k] - fields_data[k])
    return D
    
def S2TranslateTest(fields_data, n):
    shift = Lx/n
    fields_data_t = [np.copy(fields_data[0]), np.copy(fields_data[1]), np.copy(fields_data[2]), np.copy(fields_data[3])]
    
    for k in [0,3]:
        for i in range(NXH):
            fields_data_t[k][i,:] = np.conj(fields_data_t[k][i,:])
                
    for k in [1,2]:
        for i in range(NXH):
            fields_data_t[k][i,:] = -np.conj(fields_data_t[k][i,:])

    for k in range(4):
        for i in range(NXH):
            fields_data_t[k][i,:] = np.exp(2*1j*i*np.pi*(shift/Lx))*fields_data_t[k][i,:]
    D = 0
    for k in range(4):
        D = D + np.linalg.norm(fields_data_t[k] - fields_data[k])
    return D

# for i in range(2, 13):
#     h5f.create_dataset('symmetries/T' + str(i), data = TranslateTest(fields_data, i))

# h5f.create_dataset('symmetries/S1T2', data = S1TranslateTest(fields_data, 2))
# h5f.create_dataset('symmetries/S2T2', data = S2TranslateTest(fields_data, 2))
# #--------------------------------------------------------------------- #

# h5f.close()

def findsoln(f0):
    f = np.copy(f0)
    if MPI.COMM_WORLD.rank == 0:
        print('---------------------------------')
        print('--- Parameters ------------------')
        print('---------------------------------')
        print('Ra = ' + str(Ra))
        print('Pr = ' + str(Pr))
        print('Ri = ' + str(Ri))
        print('Rrho = ' + str(Rrho))
        print('Lambda = ' + str(Lambda))
        print('tau = ' + str(tau))
        print('NX = ' + str(NX))
        print('NZ = ' + str(NZ))
        print('Lz = ' + str(Lz))
        print('Lx = ' + str(Lx))
        if Tsearch:
            print('T0 = ' + str(T_guess))
        if Rxsearch:
            print('ax0 = ' + str(ax_guess))
        if Rzsearch:
            print('az0 = ' + str(az_guess))
        print('timestep = ' + str(dt_nominal))
        print('trust radius = ' + str(trust_radius))
        if symmetry:
            print('Imposed symmetry =', symm)
        print('---------------------------------', flush=True)

        log_file.writelines('---------------------------------' + '\n')
        log_file.writelines('--- Parameters ------------------' + '\n')
        log_file.writelines('---------------------------------' + '\n')
        log_file.writelines('Ra = ' + str(Ra) + '\n')
        log_file.writelines('Pr = ' + str(Pr) + '\n')
        log_file.writelines('Ri = ' + str(Ri) + '\n')
        log_file.writelines('Rrho = ' + str(Rrho) + '\n')
        log_file.writelines('Lambda = ' + str(Lambda) + '\n')
        log_file.writelines('tau = ' + str(tau) + '\n')
        log_file.writelines('NX = ' + str(NX) + '\n')
        log_file.writelines('NZ = ' + str(NZ) + '\n')
        log_file.writelines('Lz = ' + str(Lz) + '\n')
        log_file.writelines('Lx = ' + str(Lx) + '\n')
        if Tsearch:
            log_file.writelines('T0 = ' + str(T_guess) + '\n')
        if Rxsearch:
            log_file.writelines('ax0 = ' + str(ax_guess) + '\n')
        if Rzsearch:
            log_file.writelines('az0 = ' + str(az_guess) + '\n')
        log_file.writelines('timestep = ' + str(dt_nominal) + '\n')
        log_file.writelines('trust radius = ' + str(trust_radius) + '\n')
        if symmetry:
            log_file.writelines('Imposed symmetry = ' + str(symm) + '\n')
        log_file.writelines('---------------------------------' + '\n')
        log_file.writelines('\n')

    

    # print(np.shape(f[(v_begin):(v_end)]))
    error = 0
    b = applyNonLinearOperator(f)
    normb = np.linalg.norm(b)
    normf = np.linalg.norm(f)
    final_error = normb/normf if relative_tolerance else normb
    newton_iter = 0
    while newton_iter < newton_iterations:
        if final_error < tolerance:
            if MPI.COMM_WORLD.rank == 0:
                print('Tolerance threshold reached.', flush=True)
                log_file.writelines('Tolerance threshold reached.' + '\n')
            # save converged solution
            plot_solution(f)
            save_solution(f,final_error)
            save_flow_properties(f)
            break
        
        if newton_iter == 0:
            if MPI.COMM_WORLD.rank == 0:
                print("Initial error = " + str(final_error) + ', L2(f)=' + str(normf), end='')
                log_file.writelines("Initial error = " + str(final_error) + ', L2(f)=' + str(normf))
                if Tsearch:
                    print(", T = " + str(f[MY+Tsearch-1]), end='')
                    log_file.writelines(", T = " + str(f[MY+Tsearch-1]))
                if Rxsearch:
                    print(", ax = " + str(f[MY+Tsearch+Rxsearch-1]), end='')
                    log_file.writelines(", ax = " + str(f[MY+Tsearch+Rxsearch-1]))
                if Rzsearch:
                    print(", az = " + str(f[MY+Tsearch+Rxsearch+Rzsearch-1]), end='')
                    log_file.writelines(", az = " + str(f[MY+Tsearch+Rxsearch+Rzsearch-1]))
                print('\n', flush=True)
                log_file.writelines("\n")
        phi_base = []

        
        T_temp = T_guess
        ax_temp = ax_guess
        az_temp = az_guess
        if Tsearch:
            T_temp = f[MY+Tsearch-1]
        if Rxsearch:
            ax_temp = f[MY+Tsearch+Rxsearch-1]
        if Rzsearch:
            az_temp = f[MY+Tsearch+Rxsearch+Rzsearch-1]
        phi_base = phi(f[:MY], T_temp, ax_temp, az_temp)
        zn, error, tr = GMRES(f, z0, phi_base, b, krylov_dim, trust_radius)

        f = f + zn # new solution: newf = f + f_newton
        b = applyNonLinearOperator(f)
        normb = np.linalg.norm(b[:MY])
        normf = np.linalg.norm(f[:MY])
        final_error = normb/normf if relative_tolerance else normb
        newton_iter = newton_iter + 1
        plot_solution(f)
        save_solution(f,final_error)
        save_flow_properties(f)
        if MPI.COMM_WORLD.rank == 0:
            print("Iteration = " + str(newton_iter) + ",  " + "error = " + str(final_error) + ', L2(f)=' + str(normf), end='')
            log_file.writelines("Iteration = " + str(newton_iter) + ",  " + "error = " + str(final_error) + ', L2(f)=' + str(normf))
            if Tsearch:
                print(", T = " + str(f[MY+Tsearch-1]), end='')
                log_file.writelines(", T = " + str(f[MY+Tsearch-1]))
            if Rxsearch:
                print(", ax = " + str(f[MY+Tsearch+Rxsearch-1]), end='')
                log_file.writelines(", ax = " + str(f[MY+Tsearch+Rxsearch-1]))
            if Rzsearch:
                print(", az = " + str(f[MY+Tsearch+Rxsearch+Rzsearch-1]), end='')
                log_file.writelines(", az = " + str(f[MY+Tsearch+Rxsearch+Rzsearch-1]))
            print(",    Linear system error = " + str(error) + ",    trust radius = " + str(tr), flush=True)
            log_file.writelines(",    trust radius = " + str(tr) + '\n')
        if timeit.default_timer() - start > time_limit*3600:
            if MPI.COMM_WORLD.rank == 0:
                print('Time limit reached.', flush=True)
                log_file.writelines('Time limit reached.' + '\n')
            break

    runtime = timeit.default_timer() - start
    if MPI.COMM_WORLD.rank == 0:
        print('Newton solver runtime = ' + str(runtime), flush=True)
        log_file.writelines('Newton solver runtime = ' + str(runtime) + '\n')

    
    
    # save T
    if Tsearch:
        T_file = open(ecs_dir+'T.txt', 'w')
        T_file.writelines(str(f[MY+Tsearch-1]))
        T_file.close()

    # save shift speeds: ax , az
    if Rxsearch:
        ax_file = open(ecs_dir+'ax.txt', 'w')
        ax_file.writelines(str(f[MY+Tsearch+Rxsearch-1]))
        ax_file.close()
    if Rzsearch:
        az_file = open(ecs_dir+'az.txt', 'w')
        az_file.writelines(str(f[MY+Tsearch+Rxsearch+Rzsearch-1]))
        az_file.close()

    # save tolerance
    tol_file = open(ecs_dir+'tol.txt', 'w')
    tol_file.writelines(str(final_error))
    tol_file.close()

    if final_error < tolerance:
        if MPI.COMM_WORLD.rank == 0:
            print('Solver converged with tolerance ' + str(final_error), flush=True)
            log_file.writelines('Solver converged with tolerance ' + str(final_error) + '\n')

    if output_full_trajectory == 1:
        # save time-dependent solution
        if Tsearch or Rxsearch or Rzsearch:
            if MPI.COMM_WORLD.rank == 0:
                print('Saving time-dependent solution ... ', end='', flush=True)
            T_temp = T_guess
            ax_temp = ax_guess
            az_temp = az_guess
            if Tsearch:
                T_temp = f[MY+Tsearch-1]
            if Rxsearch:
                ax_temp = f[MY+Tsearch+Rxsearch-1]
            if Rzsearch:
                az_temp = f[MY+Tsearch+Rxsearch+Rzsearch-1]
            phi_out(f[:MY], T_temp, ax_temp, az_temp)
            if MPI.COMM_WORLD.rank == 0:
                if not os.path.exists(ecs_dir+'time-dependent'):
                    os.mkdir(ecs_dir+'time-dependent')
                if os.path.exists(ecs_dir+'full_solution'):
                    shutil.move(ecs_dir+'full_solution', ecs_dir+'time-dependent/full_solution')
                # if os.path.exists(ecs_dir+'order_params'):
                #     shutil.move(ecs_dir+'order_params', ecs_dir+'time-dependent/order_params')
            if MPI.COMM_WORLD.rank == 0:
                print('done', flush=True)
        

        







    if ECS_eigen:
        if final_error < tolerance:
            if MPI.COMM_WORLD.rank == 0:
                if not os.path.exists(ecs_dir+'stability/'):
                    os.mkdir(ecs_dir+'stability/')
            # compute Floquet multipliers
            
            T_temp = T_guess
            ax_temp = ax_guess
            az_temp = az_guess
            if Tsearch:
                T_temp = f[MY+Tsearch-1]
            if Rxsearch:
                ax_temp = f[MY+Tsearch+Rxsearch-1]
            if Rzsearch:
                az_temp = f[MY+Tsearch+Rxsearch+Rzsearch-1]

            phi_base = phi(f[:MY], T_temp, ax_temp, az_temp)

            # Floquet method
            Q, H_ = arnoldi_iteration(f[:MY], phi_base, T_temp, ax_temp, az_temp, np.random.rand(MY), Ne) # <-- Ne iterations
            H = H_[0:-1,:]

            # Last row of H_ (h_{m+1,m})
            h_last = H_[-1, :]  # 1 x m


            if MPI.COMM_WORLD.rank == 0:
                scipy.io.mmwrite(ecs_dir+'stability/H.mtx', H) # Hessenberg matrix

            # get eigenvalue and eigenvector results, these are Floquet multipliers
            eigenvalues, eigenvectors_ = scipy.linalg.eig(H) 
            eigenvalues_abs = np.abs(eigenvalues)
            eigenvectors = np.matmul(Q[:,0:-1], eigenvectors_)
            growthrate = np.log(eigenvalues) / T_temp # convert to growth rate

            # Residual for each Ritz pair
            res = np.zeros(eigenvalues.size)
            for i in range(eigenvalues.size):
                y = eigenvectors_[:, i]           # THIS is the eigenvector of H
                res[i] = abs(h_last @ y)

            # Sort modes by descending growth rate
            idx = np.argsort(growthrate.real)[::-1]
            growthrate = growthrate[idx]
            eigenvalues  = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            res = res[idx]

            if MPI.COMM_WORLD.rank == 0:
                # save Floquet multiplier
                scipy.io.mmwrite(ecs_dir+'stability/eigenvalues.mtx', eigenvalues.reshape(1, -1)) # Eigenvalues; .mtx = Matrix Market format
                # save growth rate
                scipy.io.mmwrite(ecs_dir+'stability/growthrate.mtx', growthrate.reshape(1, -1))
                # save residual
                scipy.io.mmwrite(ecs_dir+'stability/residual.mtx', [res])

            unstable = np.where(growthrate.real > 0)[0]
            if MPI.COMM_WORLD.rank == 0 and unstable.size > 0:
                eigenvectors_unstable = eigenvectors[:, unstable]
                eigenvalues_unstable = eigenvalues[unstable]
                growthrate_unstable = growthrate[unstable]
                h5f = h5py.File(ecs_dir+'stability/eigen_unstable.h5', 'w') 
                h5f.create_dataset('/eigenvectors', data = eigenvectors_unstable) 
                h5f.create_dataset('/eigenvalues', data = eigenvalues_unstable) 
                h5f.create_dataset('/growthrate', data = growthrate_unstable) 
                h5f.create_dataset('/xg', data = xg) 
                h5f.create_dataset('/zg', data = zg) 
                h5f.close()

            counter = unstable.size
            if MPI.COMM_WORLD.rank == 0:
                outputFile = open(ecs_dir+'stability/N_unstable.txt', 'w')
                outputFile.writelines(str(counter))
                outputFile.close()





if ecsmode is None:
    error = 0
    b = applyNonLinearOperator(f)
    final_error = np.linalg.norm(b)/np.linalg.norm(f) if relative_tolerance else np.linalg.norm(b)
    # save tolerance
    tol_file = open(ecs_dir+'tol.txt', 'w')
    tol_file.writelines(str(final_error))
    tol_file.close()
else:
    findsoln(f)

stop = timeit.default_timer()
if MPI.COMM_WORLD.rank == 0:
    print('Time: ' + str(stop-start) + ' seconds')
    log_file.writelines('Time: ' + str(stop-start) + ' seconds')
    log_file.close()

MPI.COMM_WORLD.Barrier()
sys.exit(0)