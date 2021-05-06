"""

Hybridizable Interior Penalty method for Solving 
Convection Diffusion problem

div(-kappa grad(u) + beta u) = f    in Omega
                           u = 0    on Gamma

kappa : Homogeneous dispersion tensor
beta  : Velocity field
f     : Source term

Copyright ©
Author: Grégory ETANGSALE, University of La Réunion

"""

##################################################################

# Import Netgen/NGSolve and Python modules

from ngsolve import *
from netgen.geom2d import unit_square
import math

ngsglobals.msg_level = 1

##################################################################

# Mesh generation

geometry = SplineGeometry()
mesh = Mesh(unit_square.GenerateMesh(maxh=1/10, quad_dominated=True))

##################################################################

# Physical parameters : kappa, beta

Identity = CoefficientFunction((1,0,0,1),dims=(2,2))

ki = 5e-1
kappa = ki * Identity
beta = CoefficientFunction( (2,1) )

# Exact solution / Flux / Source term

A = (y+(exp(y/ki)-1)/(1-exp(1/ki)))
B = (exp(2*x/ki)-1)/(1-exp(2/ki))
C = (x+(exp(2*x/ki)-1)/(1-exp(2/ki)))
D = (exp(y/ki)-1)/(1-exp(1/ki))

exact = CoefficientFunction( C * A )
f = CoefficientFunction( 2 * A + C )
flux = CoefficientFunction( (A*(1+(2/ki)*B) , C*(1+(1/ki)*D)) )

Draw(exact, mesh, "exact_solution")

##################################################################

# Physical parameters of the H-IP method

order = 1                       # polynomial degree

gamma = 2*(order+1)*(order+2)   # constant for stabilization function
epsilon = 0                     # variant of the H-IP formulation

theta = 1                       # value for the theta-scheme
SG = True                       # Upwind or Scharfetter-Gummel function (True for SG / False for Upwind)

condense = True                 # static condensation 

##################################################################

# Main functions

def FES():
    # Finite Element Spaces
    V    = L2(mesh, order=order)
    Vhat = FacetFESpace(mesh, order=order, dirichlet="left|right|top|bottom")
    fes  = FESpace([V,Vhat], dgjumps=True)
    
    print ("vdofs:    ", fes.Range(0))
    print ("vhatdofs: ", fes.Range(1))

    return fes


def stabilization_function(n,h):
    Kn = InnerProduct( n, CoefficientFunction(kappa*n,dims=(2,1)) )

    if SG: # Scharfetter-Gummel
        tau_dif = gamma*Kn/h
        Pe = theta*beta*n / tau_dif
        tau = tau_dif * ( -sqrt(Pe*Pe)/(exp(-sqrt(Pe*Pe))-1) )
    else: # Additive Upwind
        Nbeta = sqrt(beta[0]**2 + beta[1]**2)
        tau = gamma*Kn/h + theta*Nbeta

    return tau


def Assembling(fes):
    u, uhat = fes.TrialFunction()
    v, vhat = fes.TestFunction()

    n = specialcf.normal(mesh.dim)      # unit normal
    h = specialcf.mesh_size             # element size
    
    a = BilinearForm(fes, eliminate_internal=condense)

    a_int = kappa*grad(u)*grad(v) - u*(beta*grad(v))
    a_fct = (kappa*grad(u)*n-u*beta*n)*(vhat-v)+epsilon*(kappa*grad(v)*n)*(uhat-u)
    a_sip = stabilization_function(n,h)*(u-uhat)*(v-vhat)
    
    a += SymbolicBFI( a_int )
    a += SymbolicBFI( a_fct + a_sip , element_boundary=True )

    l = LinearForm(fes)
    l += SymbolicLFI(f*v)

    gfu = GridFunction(fes)

    return a,l,gfu


def compute_L2_error(uh):
    return sqrt( Integrate((uh-ue)**2, mesh, order=2*order ) )

##################################################################

# Main programm

fes = FES()
[a,l,gfu]=Assembling(fes)

a.Assemble()
l.Assemble()

# Solve Linear System
if condense:
    l.vec.data += a.harmonic_extension_trans * l.vec

    inv = a.mat.Inverse(fes.FreeDofs(True), "umfpack")
    gfu.vec.data = inv * l.vec

    gfu.vec.data += a.harmonic_extension * gfu.vec
    gfu.vec.data += a.inner_solve * l.vec
else:
    inv = a.mat.Inverse(fes.FreeDofs(), "umfpack")
    gfu.vec.data = inv * l.vec

uh = gfu.components[0]
L2_error = compute_L2_error(uh)

print("L2-norm estimate = ", L2_error)
Draw(uh,mesh,"solution")

##################################################################
