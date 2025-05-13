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
        mesh_path='/scratch/s/steinman/ranbar/Torino/Coronary/mesh/',
        mesh_name='MildStenosis_mesh.xml',
        BC_file_name='MildStenosis_BCnodesFacets.xml',
        save_step=10, #10,
        folder='results/',
		linear_solver="mumps",
        checkpoint=100,
        plot_interval=200, #100,
        print_intermediate_info=1e7,
        print_velocity_pressure_convergence=True,
        use_krylov_solvers=True)
    
    
    # Obtain marked boundaries
    #id_in  = read_mesh_info(mesh_info, mesh_path, '<INLETS>')
    #id_out = read_mesh_info(mesh_info, mesh_path, '<OUTLETS>')
    #NS_parameters.update(id_in=id_in, id_out=id_out)

    # Calculate inflow velocity profile
    #>>>>> inflow_Vprof = get_inflow_Vprofile(inlet_diameter)
    
    # Update these in NS parameters and expression
    #NS_parameters.update(ave_inlet_velocity= average_inlet_velocity)
    #>>>>> NS_expressions.update(dict(u_in= inflow_Vprof)) 



# ---------------------------- Read Mesh  -----------------------------#

# all length scales are based on the mesh (in [meters] for most cases)
def mesh(mesh_name, mesh_path, **NS_namespace):
    # to read .xml mesh
    mesh = Mesh(mesh_path + mesh_name)

    # to read .h5 mesh
    #h5   = HDF5File(mesh.mpi_comm(), mesh_path + mesh_name, "r")
    #h5.read(mesh, "/mesh", False)

    return mesh


# ---------------------------- Boundary Conditions -----------------------------#

# ----------------------- Extract subdomains -------------------------------
def get_subdomains(mesh, **NS_namespace):

    # Create an empty MeshFunction on facets (dim-1)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    # Read the raw indices & values from the HDF5
    #mesh_h5  = HDF5File(mesh.mpi_com(), mesh_path + mesh_name, "r")
    #mesh_h5.read(sub_domains, "face_marker")
    #indices  = f["/facet_marker/indices"][:]  # shape (F,3)
    #values   = f["/facet_marker/values"][:]   # shape (F,)
    #mesh_h5.close()

    return sub_domains


def create_bcs(V, Q, mesh, sub_domains, **NS_namespace):

    # walls: sub_domains(1), inlet: sub_domains(3), outlet: sub_domains(2)
    sub_domains = get_subdomains(mesh, mesh_name, mesh_path, **NS_namespace)

    # Velocity BCs

    # 1. Noslip condition at wall  
    noslip = Constant(0.0)
    bcu_walls  = DirichletBC(V, noslip, sub_domains, 1) 

    # 2. Inlet velocity (constant parabola)
    bcux_inlet = DirichletBC(V, noslip, sub_domains, 3) #DirichletBC(V, NS_expressions['u_in'], sub_domains, 3) #inlet velocity
    bcuy_inlet = DirichletBC(V, noslip, sub_domains, 3)
    bcuz_inlet = DirichletBC(V, noslip, sub_domains, 3)

    #Pressure BCs: Dirichlet for outlet pressure -> sets a reference pressure
    bcp_outlet = DirichletBC(Q, 0, sub_domains, 2) 

    return dict(u0= [bcu_walls, bcux_inlet], #x-component of velocity -> 0 on the wall
                u1= [bcu_walls, bcuy_inlet], #y-component of velocity -> 0 on wall and inlet
                u2= [bcu_walls, bcuz_inlet], #z-component of velocity -> 0 on wall and inlet
                p = [bcp_outlet])            #Prescribed inlet, zero outlet
    

"""
def get_inlet_center(mesh, sub_domains, **NS_namespace):
    inlet = sub_domains(3)


def get_inflow_Vprofile(diameter, mesh, sub_domains **NS_namespace):
    #diameter = 0.02 #[m]
    R = diameter/2
    area = np.pi * R**2

    # Total inlet flow rate
    Q_inflow = 1.43 * diameter**2.55 

    # max inlet velocity (based on Poiseuille flow)
    umax = 2*Q_inflow/area

    # Get coordinates of center
    center = get_inlet_center(mesh, sub_domains)
    y_c, z_c = center[1], center[2] 

    # Calculate distance of points from center
    radius = np.sqrt((x[1] - y_c)**2 + (x[2] - z_c)**2)

    #Impose steady profile (Poiseuille parabola)
    inflow_Vprof = Expression('umax*(1- pow(r/R,2))', umax= umax, r= radius, R= R, degree=2)

    return inflow_Vprof

"""




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



