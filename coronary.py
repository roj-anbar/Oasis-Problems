"""
To solve the incompressible Navier-Stokes equations for 3D coronary cases (with mild and severe stenosis) using oasis.

The mesh was provided from Torino group.
Prescribed velocity boundary conditions due to venous inflow rates provided by Torino group.


@Author: Rojin Anbarafshan
Contact: rojin.anbar@gmail.com
Date: May 2025
"""

from __future__ import print_function
from oasis import *
from oasis.problems import *
from oasis.problems.NSfracStep import *
from oasis.common import * #added to be able to use io.py functions
from dolfin import *
import numpy as np
import sys
from ufl import dot, grad
from dolfin import project, Function, XDMFFile


# ---------------------------- Problem Setup -----------------------------#
def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):


    NS_parameters.update(
        nu= 0.0035, #[Pa.s]
        Re=1,
        period=1,
        dt=0.001, 
        inlet_area = 7.55527*1e-4, #[m2]
        inlet_centroid = [-0.216742, 0.161571, -0.29505], #center of mass [m]
        mesh_path='/scratch/s/steinman/ranbar/Torino/Coronary/mesh/',
        mesh_name='MildStenosis_mesh.xml',
        BC_file='MildStenosis_BCnodesFacets.xml',
        save_step=10, #10,
        folder='results/',
		linear_solver="mumps",
        checkpoint=100,
        plot_interval=200, #100,
        print_intermediate_info=1e7,
        print_velocity_pressure_convergence=True,
        use_krylov_solvers=True)


    # Calculate inflow velocity profile
    inflow_Vprof = get_inflow_Vprofile(NS_parameters['inlet_centroid'], NS_parameters['inlet_area'],)
    # Update this in NS parameters and expression
    NS_expressions.update(dict(u_in= inflow_Vprof)) 
    
    # Placeholder for u_in
    #NS_expressions["u_in"] = Constant(0.0) 




# ---------------------------- Read mesh and obtain useful info -----------------------------#

# all length scales are based on the mesh (in [meters] for most cases)
def mesh(mesh_name, mesh_path, **NS_namespace):
    # Load the volume mesh .xml
    mesh = Mesh(mesh_path + mesh_name)

    # to read .h5 mesh
    #h5   = HDF5File(mesh.mpi_comm(), mesh_path + mesh_name, "r")
    #h5.read(mesh, "/mesh", False)

    return mesh

"""
def get_inlet_params(mesh_path, inlet_vtp_file, **NS_namespace):
    
    #Compute the centroid (center of mass) and area of the inlet by reading the inlet.vtp patch.
    #mesh coords are assumed in [cm]; returns (center, area) in [m] and [m2].
   
    # Collect all unique vertex‐indices on inlet facets
    inlet_points = set()
    for facet in facets(mesh):
        if sub_domains[facet.index()] == inlet_tag:
            for pid in facet.entities(0):
                inlet_points.add(pid)

    # Get coordinates of all inlet points and average
    coords        = mesh.coordinates()*0.01           # (N_points × 3) array  -> converts from [cm] to [m]
    inlet_coords  = coords[list(inlet_points), :]     # (#inlet_points × 3)
    center        = inlet_coords.mean(axis=0)         # length‑3 (gdim) array
    

    # center = (x_c, y_c, z_c) in [meters]
    #center = center_cm * 0.01

    # Compute radial distances and diameter in [meters]
    radii        = np.linalg.norm(inlet_coords - center, axis=1)
    diameter     = 2.0 * radii.max()
    

    print('Inlet center [m]:', center)
    
    return center

"""                

    

# ---------------------------- Boundary Conditions -----------------------------#


def create_bcs(V, Q, mesh, mesh_path, BC_file, **NS_namespace):
    """
    Use the get_sub_domains() to obtain boundaries and then apply Dirichlet BCs on them as:
      • walls  (tag 1)   → no slip
      • inlet  (tag 2)   → prescribe inlet velocity (u_in)
      • outlet (tag 3)   → p = 0
    """
    wall_tag   = 1
    inlet_tag  = 2
    outlet_tag = 3

    sub_domains = get_subdomains(mesh, mesh_path, BC_file, **NS_namespace)
    
    #inlet_diameter = NS_parameters['inlet_diameter']
    #inflow_Vprof = get_inflow_Vprofile(mesh, sub_domains)
    
    # Update these in NS parameters and expression
    #NS_expressions.update(dict(u_in= inflow_Vprof)) 

    # Velocity BCs
    bcu = [[],[],[]]

    # 1. Noslip condition at wall  
    noslip = Constant(0.0)
    bcu_walls  = DirichletBC(V, noslip, sub_domains, wall_tag) # this is for each velocity component (and it should hold for all 3 directions)

    # 2. Inlet velocity (constant parabola)
    bcux_inlet = DirichletBC(V, NS_expressions['u_in'], sub_domains, inlet_tag)
    bcuy_inlet = DirichletBC(V, noslip, sub_domains, inlet_tag)
    bcuz_inlet = DirichletBC(V, noslip, sub_domains, inlet_tag)


    # Combine both velocity BCs
    bcu[0] = [bcu_walls, bcux_inlet]
    bcu[1] = [bcu_walls, bcuy_inlet]
    bcu[2] = [bcu_walls, bcuz_inlet]


    #Pressure BCs: Dirichlet for outlet pressure -> sets a reference pressure
    outflow = Constant(0.0) #Expression("p", p = 0, degree=pressure_degree)
    bcp_outlet = DirichletBC(Q, outflow, sub_domains, outlet_tag) 

    return dict(u0= bcu[0],  #x-component of velocity -> 0 on the wall
                u1= bcu[1],  #y-component of velocity -> 0 on wall and inlet
                u2= bcu[2],  #z-component of velocity -> 0 on wall and inlet
                p = [bcp_outlet])    #Prescribed inlet, zero outlet
    



# Extract subdomains 
def get_subdomains(mesh, mesh_path, BC_file, **NS_namespace):
    """
    Load the facet‐marker MeshFunction from the boundary XML file containing boundary tags:
    1=wall, 2=inlet, 3=outlet
    """

    # Create MeshFunction on facets (dim-1)
    #sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1, mesh_path+BC_file)
    sub_domains = MeshFunction("size_t", mesh, mesh_path+BC_file)

    return sub_domains


def get_inflow_Vprofile(center, area, **NS_namespace):
    
    # Get area and centroid of the inlet
    #center, area = get_inlet_params(mesh, sub_domains) #[-21.6647, 16.2042, -29.5128]

    #center = NS_parameters["inlet_centroid"]
    #area = NS_parameters["inlet_area"]

    diameter = sqrt(4*area/np.pi)
    radius = diameter/2
    
    xc, yc, zc = center[0], center[1], center[2] 
    
    # Total inlet flow rate
    Q_inflow = 1.43 * diameter**2.55 

    # max inlet velocity (based on Poiseuille flow)
    umax = 2*Q_inflow/area

    # Calculate distance of points from center
    #r = np.sqrt((x[1] - y_c)**2 + (x[2] - z_c)**2)

    #Impose steady profile (Poiseuille parabola)
    uin_expr = ('umax*(1 - pow(sqrt((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc) + (x[2]-zc)*(x[2]-zc))/R, 2) )')
    inflow_Vprof = Expression(uin_expr, umax= umax, xc = xc, yc= yc, zc= zc, R= radius, degree=2)

    print('Q_inflow [m3/s] =', Q_inflow)
    return inflow_Vprof






# ---------------------------- Oasis Functions -----------------------------#
# only called once, before the first time‐step
def pre_solve_hook(mesh, u_, p_, AssignedVectorFunction, **NS_namespace):
    
    uv = AssignedVectorFunction(u_, "Velocity")
    normals = FacetNormal(mesh)

    #return dict(uv= AssignedVectorFunction(u_, "Velocity"), n= FacetNormal(mesh))
    return dict(uv=uv, normals=normals)



def start_timestep_hook(t, u_in, **NS_namespace): #originally it was P_in instead of u_in
   #u_in.t = t
   #P_in.t = t
   pass
    

def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass


# Called at each time-step
def temporal_hook(u_, p_, mesh, tstep, uv, normals, print_intermediate_info, plot_interval, **NS_namespace):
    
    # I/O
    if tstep % print_intermediate_info == 0:
        #print("Continuity ", assemble(dot(u_, n) * ds()))
        pinlet = p_.get_local()
        print("pressure gradient: ", pinlet)

    if tstep % plot_interval == 0:
        u_max = max(u_[0].vector().get_local().max(), u_[1].vector().get_local().max())
        #print("tstep= ", tstep, "and umax=", u_max)
        print("tstep= ", tstep, ": worst Courant", NS_parameters['dt']*u_max/mesh.hmin())



def theend_hook(uv, p_, **NS_namespace):
    uv()
    #plot(uv, title='Velocity')
    #plot(p_, title='Pressure')



