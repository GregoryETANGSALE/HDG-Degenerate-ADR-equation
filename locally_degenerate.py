"""

Hybridizable Interior Penalty method for Solving 
Locally degenerate Advection-Diffusion-Reaction problem

div(-kappa grad(u) + beta u) + gamma u = f     in Omega
                                     u = uD    on GammaD

kappa : Homogeneous dispersion tensor
beta  : Velocity field
gamma : reaction coefficient
f     : Source term

Copyright ©
Author: Grégory ETANGSALE, University of La Réunion

"""

##################################################################

# Import Netgen/NGSolve and Python modules

from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters, meshsize

import math

ngsglobals.msg_level = 1

##################################################################

# Mesh generation

h1 = 1
h2 = 1
h3 = 1

def lowerpart1(geom):
    lower_pnts = [ (-1,-1), (0,-1), (1,-1), (1,0), (0.5,0), (0.5,-0.5), (0,-0.5), (-0.5,-0.5), (-0.5,0), (-1,0) ]
    lower_nums = [geom.AppendPoint(*p) for p in lower_pnts]
    
    lines = [ (lower_nums[0], lower_nums[1], "gammaOut", 1, 0, h1),
              (lower_nums[1], lower_nums[6], 12, 1, 2, h3),
              (lower_nums[8], lower_nums[9], 11        , 1, 4, h3),
              (lower_nums[9], lower_nums[0], "gammaIn1", 1, 0, h1)]

    for p0,p1,bc,left,right,h in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=h)

    splines = [ (lower_nums[6], lower_nums[7], lower_nums[8], "gammaIn1", 1, 0,h2)]

    for p0,p1,p2,bc,left,right,h in splines:
        geom.Append( ["spline3", p0, p1, p2], bc=bc, leftdomain=left, rightdomain=right,maxh=h)

    geom.SetMaterial(1, "Ldomain1")
    
    return (geom, lower_nums)


def lowerpart2(geom,lower_nums):
    lines = [ (lower_nums[1], lower_nums[2], "gammaIn2", 2, 0, h1),
              (lower_nums[2], lower_nums[3], "gammaOut", 2, 0, h1),
              (lower_nums[3], lower_nums[4], 23 , 2, 3, h3)]

    for p0,p1,bc,left,right,h in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=h)

    splines = [ (lower_nums[4], lower_nums[5], lower_nums[6], "gammaIn2", 2, 0, h2)]

    for p0,p1,p2,bc,left,right,h in splines:
        geom.Append( ["spline3", p0, p1, p2], bc=bc, leftdomain=left, rightdomain=right,maxh=h)

    geom.SetMaterial(2, "Ldomain2")
    return (geom, lower_nums)


def upperpart1(geom, lower_nums):
    upper_pnts = [ (1,1), (0,1), (-1,1), (0.5,0.5), (0,0.5), (-0.5,0.5) ]
    upper_nums = [geom.AppendPoint(*p) for p in upper_pnts]
    lines = [(lower_nums[3], upper_nums[0], "gammaD1", 3, 0, h1),
             (upper_nums[0], upper_nums[1], "gammaD1", 3, 0, h1),
             (upper_nums[1], upper_nums[4], 34, 3, 4, h3)]

    for p0,p1,bc,left,right,h in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=h)
        
    splines = [(upper_nums[4], upper_nums[3], lower_nums[4], "gammaD1", 3, 0, h2)]

    for p0,p1,p2,bc,left,right,h in splines:
        geom.Append( ["spline3", p0, p1, p2], bc=bc, leftdomain=left, rightdomain=right,maxh=h)

    geom.SetMaterial(3, "Udomain1")

    return (geom, upper_nums)



def upperpart2(geom, lower_nums, upper_nums):
    lines = [(upper_nums[1], upper_nums[2], "gammaD2", 4, 0, h1),
             (upper_nums[2], lower_nums[9], "gammaD2", 4, 0, h1)]

    for p0,p1,bc,left,right,h in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=h)
        
    splines = [(lower_nums[8], upper_nums[5], upper_nums[4], "gammaD2", 4, 0, h2)]

    for p0,p1,p2,bc,left,right,h in splines:
        geom.Append( ["spline3", p0, p1, p2], bc=bc, leftdomain=left, rightdomain=right,maxh=h)

    geom.SetMaterial(4, "Udomain2")

    return (geom)

def MakeMesh():
    geometry = SplineGeometry()
    geometry, lower = lowerpart1(geometry)
    geometry, lower = lowerpart2(geometry,lower)
    geometry, upper = upperpart1(geometry, lower)
    geometry = upperpart2(geometry, lower, upper)
     
    return Mesh(geometry.GenerateMesh(meshsize.very_coarse, curvaturesafety=0, quad_dominated=False))    

mesh = MakeMesh()
mesh.Curve(1)

Draw(mesh)
mesh.Refine()
Redraw()
mesh.Refine()
Redraw()
mesh.Refine()
Redraw()

##################################################################

# Physical parameters : kappa, beta, gamma

Identity = CoefficientFunction((1,0,0,1),dims=(2,2))

kappal = (0,0,0,0)
kappau = (pi,0,0,pi)
kappar = { "Udomain1" : kappau, "Udomain2" : kappau, "Ldomain1" : kappal, "Ldomain2" : kappal }
kappa_coef = [ kappar[mat] for mat in mesh.GetMaterials() ]
kappa = CoefficientFunction(kappa_coef,dims=(2,2))

beta = CoefficientFunction(( -y/(x*x+y*y), x/(x*x+y*y) ))

mu = 1e-12

# Exact solution / Source term

# Variable alpha (polar coordinates)
d = 1e-15
alphaU1 = atan((y+d)/(x+d))
alphaU2 = atan((y+d)/(x-d))+pi
alphaL1 = atan((y-d)/(x-d))+pi
alphaL2 = atan((y-d)/(x+d))+2.0*pi

alpha_mat = { "Udomain1" : alphaU1, "Udomain2" : alphaU2, "Ldomain1" : alphaL1, "Ldomain2" : alphaL2 }
alpha_coef = [ alpha_mat[mat] for mat in mesh.GetMaterials() ]
alpha = CoefficientFunction(alpha_coef)

Draw(alpha,mesh,"theta") 

# Exact solution
exactU = (alpha-pi)*(alpha-pi)
exactL = 3.0*pi*(alpha-pi)

exact_mat = { "Udomain1" : exactU, "Udomain2" : exactU, "Ldomain1" : exactL, "Ldomain2" : exactL }
exact_coef = [ exact_mat[mat] for mat in mesh.GetMaterials() ]
exact = CoefficientFunction(exact_coef)
Draw(exact, mesh, 'exact')

# Flux
rr = sqrt(x*x + y*y)

duxU = -(2*(alpha-pi)/rr)*sin(alpha)
duxL = -(3*pi/rr)*sin(alpha)
dux_mat = { "Udomain1" : duxU, "Udomain2" : duxU, "Ldomain1" : duxL, "Ldomain2" : duxL }
dux_coef = [ dux_mat[mat] for mat in mesh.GetMaterials() ]

duyU = (2*(alpha-pi)/rr)*cos(alpha)
duyL = (3*pi/rr)*cos(alpha)
duy_mat = { "Udomain1" : duyU, "Udomain2" : duyU, "Ldomain1" : duyL, "Ldomain2" : duyL }
duy_coef = [ duy_mat[mat] for mat in mesh.GetMaterials() ]

flux = CoefficientFunction( ( dux_coef , duy_coef ) )
Draw(flux,mesh,"velocity")

# Source Terme
fU = (2.0/(rr*rr))*(alpha-2.0*pi) + mu*(alpha-pi)*(alpha-pi)
fL = 3.0*pi*( (1.0/(rr*rr)) + mu*(alpha-pi))

f_mat = { "Udomain1" : fU, "Udomain2" : fU, "Ldomain1" : fL, "Ldomain2" : fL }
f_coef = [ f_mat[mat] for mat in mesh.GetMaterials() ]
f = CoefficientFunction(f_coef)

##################################################################

# Numerical parameters of the H-IP method

order = 4                       # polynomial degree
epsilon = 1                     # variation of the H-IP scheme
gamma = 2*(order+1)*(order+2)   # constant for stabilization function
SG = True                       # Upwind or Scharfetter-Gummel function (True for SG / False for Upwind)

condense = True                 # static condensation 

##################################################################

# Main functions

def FES():
    # Selecting interior polynomial basis for u (V) and facet polynomial basis for ubar (Vhat)
    V    = L2(mesh, order=order)
    Vhat = FacetFESpace(mesh, order=order, dirichlet="gammaD1|gammaD2|gammaIn1|gammaIn2")
    # Primal HDG : Create the associated Finite Element space
    fes  = FESpace([V,Vhat], dgjumps=True)
    
    print ("vdofs:    ", fes.Range(0))
    print ("vhatdofs: ", fes.Range(1))

    return fes


def stabilization_function(n,h):
    # Definition of normal diffusivity
    Kn = InnerProduct( n, CoefficientFunction(kappa*n,dims=(2,1)) )
    tau_dif = gamma*Kn/h
    Pe = (beta*n) / tau_dif + IfPos(beta*n, 1, 1)*1e-12

    # Definition of IP Stabilisation (tau) parameters used in our HDG method
    if SG: # Scharfetter-Gummel
        tau = tau_dif * ( -sqrt(Pe*Pe)/(exp(-sqrt(Pe*Pe))-1) )
    else: # Additive Upwind
        tau = tau_dif * ( 1 + sqrt(Pe*Pe) )

    return tau


def Assembling(fes):
    # Primal HDG : Create the associated discrete variables u & uhat
    u, uhat = fes.TrialFunction()
    v, vhat = fes.TestFunction()

    n = specialcf.normal(mesh.dim)      # unit normal
    h = specialcf.mesh_size             # element size
    
    # Creating the bilinear form associated to our primal HDG method:
    a = BilinearForm(fes, eliminate_internal=condense)

    # 1. Interior terms
    a_int = InnerProduct(kappa*grad(u),grad(v)) - u*(beta*grad(v)) + mu*u*v
    # 2. Boundary terms : residual
    a_fct = (kappa*grad(u)*n-u*beta*n)*(vhat-v)+epsilon*(kappa*grad(v)*n)*(uhat-u)
    a_sta_ell = stabilization_function(n,h)*(u-uhat)*(v-vhat)
    a_sta_hyp = (IfPos(beta*n, 0, -1)*beta*n+IfPos(beta*n, 0, 1)*1e-12)*(u-uhat)*(v-vhat)
    # 3. Outflow terms : only on the outflow part of the boundary
    a_outflow = sqrt((beta*n)*(beta*n))*uhat.Trace()*vhat.Trace()

    a += SymbolicBFI( a_int )
    a += SymbolicBFI( a_fct, element_boundary=True )
    a += SymbolicBFI( a_sta_ell, element_boundary=True, definedon=mesh.Materials("Udomain1|Udomain2") )
    a += SymbolicBFI( a_sta_hyp, element_boundary=True, definedon=mesh.Materials("Ldomain1|Ldomain2") )
    a += SymbolicBFI( a_outflow, definedon=mesh.Boundaries("gammaOut") )

    # Creating the linear form associated to our primal HDG method:
    l = LinearForm(fes)
    l += SymbolicLFI(f*v)

    # gfu : total vector of dof [u,uhat]
    gfu = GridFunction(fes)

    # Some options for the solver
    #c = Preconditioner(a, "local")
    c = Preconditioner(type="direct", bf=a, inverse="umfpack")
    #c = Preconditioner(a, type="multigrid", flags= { "inverse" : "sparsecholesky" })

    return a,l,gfu,c


def compute_L2_error(uh):
    # CalcL2Error computes the L2 error of the discrete variable ||u-uh||_X=sqrt(sum(||u-uh||^2_A))
    return sqrt( Integrate((uh-exact)**2, mesh, order=2*order ) )

##################################################################

# Main programm

def SolveBVP():
    fes = FES()
    fes.Update()
    [a,l,gfu,c]=Assembling(fes)
    gfu.Update()

    # Impose boundary conditions
    alpha = (
        IfPos(x, 1, 0)*IfPos(y, 1, 0)*atan((y+d)/(x+d))
        + IfPos(x, 0, 1)*IfPos(y, 1, 0)*(atan((y+d)/(x-d))+pi)
        + IfPos(x, 0, 1)*IfPos(y, 0, 1)*(atan((y-d)/(x-d))+pi)
        + IfPos(x, 1, 0)*IfPos(y, 0, 1)*(atan((y-d)/(x+d))+2.0*pi)
    )
    ubndD1  = (alpha-pi)*(alpha-pi)
    ubndD2  = (alpha-pi)*(alpha-pi)
    ubndIn1 = 3.0*pi*(alpha-pi)
    ubndIn2 = 3.0*pi*(alpha-pi)

    ue = CoefficientFunction(
        IfPos(x, 0, 1)*IfPos(y, 0, 1)*ubndIn1
        + IfPos(x, 1, 0)*IfPos(y, 0, 1)*ubndIn2
        + IfPos(x, 1, 0)*IfPos(y, 1, 0)*ubndD1
        + IfPos(x, 0, 1)*IfPos(y, 1, 0)*ubndD2 ) 
     
    gfu.components[1].Set(ue, definedon=mesh.Boundaries("gammaIn1|gammaIn2|gammaD1|gammaD2"))

    # Assembly of bilinear a() and linear f() terms
    a.Assemble()
    l.Assemble()

    # Solve the Linear System (with or without static condensation)
    BVP(bf=a, lf=l, gf=gfu, pre=c, maxsteps=3, prec=1.0e-16).Do()

    # compute L2-norm error
    uh = gfu.components[0]
    Draw(uh,mesh,"solution")

    L2_error = compute_L2_error(uh)
    print("L2-norm estimate")
    print(fes.ndof,",",mesh.ne,",",L2_error)



#while mesh.ne < 1000:
#    #print ("Number of elements:", mesh.ne)
#    SolveBVP()
#    mesh.Refine()

SolveBVP()

##################################################################







