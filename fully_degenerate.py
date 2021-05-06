"""

Hybridizable Interior Penalty method for Solving 
Convection problem

div(beta u) = f    in Omega
          u = 0    on Gamma

beta  : Velocity field
f     : Source term

Copyright ©
Author: Grégory ETANGSALE, University of La Réunion

"""

##################################################################

# Import Netgen/NGSolve and Python modules

from ngsolve import *
from netgen.geom2d import SplineGeometry

ngsglobals.msg_level = 1

##################################################################

# Mesh generation

def domain(geom):
    coord = [ (0,0), (1,0), (1,1), (0,1), (0,0.5) ]
    nums1 = [geom.AppendPoint(*p) for p in coord]
    lines = [(nums1[0], nums1[1], "gammaD0" , 1, 0),
             (nums1[1], nums1[2], "gammaOut", 1, 0),
             (nums1[2], nums1[3], "gammaOut", 1, 0),
             (nums1[3], nums1[4], "gammaD1" , 1, 0),
             (nums1[4], nums1[0], "gammaD0" , 1, 0)]

    for p0,p1,bc,left,right in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=1/2)

    geom.SetMaterial(1, "domain")
    return (geom)

geo = SplineGeometry()
geo = domain(geo)
mesh = Mesh(geo.GenerateMesh(maxh=1/2,quad_dominated=False))

##################################################################

# Physical parameters : beta
beta  = CoefficientFunction( (2, 1) )

# Exact solution / Source term
exact = CoefficientFunction( IfPos(-x+2*y-1, 1, 0)*(1+0.0*x) )
f = CoefficientFunction( 0 )

Draw(exact, mesh, "exact_solution")

##################################################################

# Physical parameters of the H-IP method

order = 1                       # polynomial degree
theta = 1                       # value for the theta-scheme
SG = True                       # Upwind or Scharfetter-Gummel function (True for SG / False for Upwind)

condense = True                 # static condensation 

##################################################################

# Main functions

def FES():
    # Selecting interior polynomial basis for u (V) and facet polynomial basis for ubar (Vhat)
    V    = L2(mesh, order=order)
    Vhat = FacetFESpace(mesh, order=order, dirichlet="gammaD0|gammaD1")
    # Primal HDG : Create the associated Finite Element space
    fes  = FESpace([V,Vhat], dgjumps=True)
    
    print ("vdofs:    ", fes.Range(0))
    print ("vhatdofs: ", fes.Range(1))

    return fes


def Assembling(fes):
    # Primal HDG : Create the associated discrete variables u & uhat
    u, uhat = fes.TrialFunction()
    v, vhat = fes.TestFunction()

    n = specialcf.normal(mesh.dim)      # unit normal
    h = specialcf.mesh_size             # element size
    
    # Creating the bilinear form associated to our primal HDG method:
    a = BilinearForm(fes, eliminate_internal=condense)

    # Definition of Stabilisation function (tau) used in our HDG method (theta-upwind scheme)
    tau = theta*sqrt((beta*n)*(beta*n))

    # 1. Interior terms
    a_int = - u*(beta*grad(v))
    # 2. Boundary terms : residual
    a_fct = (-u*beta*n)*(vhat-v)
    a_sta = tau*(u-uhat)*(v-vhat)
    # 3. Outflow terms : only on the outflow part of the boundary
    a_outflow = beta*n*uhat.Trace()*vhat.Trace()
    
    a += SymbolicBFI( a_int )
    a += SymbolicBFI( a_fct + a_sta , element_boundary=True )
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


# CalcL2Error computes the L2 error of the discrete variable ||u-uh||_X=sqrt(sum(||u-uh||^2_A))
def compute_L2_error(uh):
    return sqrt( Integrate((uh-exact)**2, mesh, order=2*order ) )

def LocalL2Error(uh):
    return Integrate( (uh-exact)**2, mesh, order=2*order, element_wise=True)


##################################################################

# Main programm

def SolveBVP():
    fes = FES()
    fes.Update()
    [a,l,gfu,c]=Assembling(fes)
    gfu.Update()

    # Impose boundary conditions
    gfu.components[1].Set(1, definedon=mesh.Boundaries("gammaD1"))

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

    vec_L2_error = LocalL2Error(uh)
    max_L2_error = max(vec_L2_error)
        
    for el in mesh.Elements():
        mesh.SetRefinementFlag(el, vec_L2_error[el.nr] > 0.05*max_L2_error)



while mesh.ne < 1000:
    #print ("Number of elements:", mesh.ne)
    SolveBVP()
    mesh.Refine()

SolveBVP()

##################################################################

