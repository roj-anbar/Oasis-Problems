"""
To solve the incompressible Navier-Stokes equations for 3D coronary cases (with mild and severe stenosis) using oasis.

The mesh was provided by the Torino group.
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
#from dolfin import UserExpression, Expression

# ---------------------------- Problem Setup -----------------------------#
def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):

    NS_parameters.update(

        # Physical constants
        nu = 1, #0.0035/1060, #[m2/s] #kinematic viscosity
        period = 1, #[s]
        dt = 0.0001, #[s]
        inlet_area = 7.5708*1e-6, #[m2] #obtained from Paraview
        inlet_centroid = [-0.0226278, 0.0164909, -0.0285813],  #center of mass [m] #obtained from Paraview

        # Input parameters
        mesh_path = '/scratch/s/steinman/ranbar/Torino/Coronary/mesh/',
        mesh_name = 'MildStenosis_New_mesh.xml',        
        BC_file = 'MildStenosis_New_BCnodesFacets.xml', 

        # Output parameters
        folder = 'results/',
        save_step = 10, #10,
        checkpoint = 100,
        #plot_interval = 200, #100,
        print_intermediate_info = 1e7,
        print_velocity_pressure_convergence = True,
        
        # Solver parameters
        linear_solver = "mumps",
        use_krylov_solvers = True
        #krylov_solvers = dict(
        #    monitor_convergence = True,
        #    report = True,
        #    maximum_iterations = 200,
        #    absolute_tolerance = 1e-6)
        )

    print("Starting simulations for dt = ", NS_parameters['dt'], '[s]')

    # Calculate inflow velocity profile
    #inflow_Vprof = get_inflow_Vprofile(NS_parameters['inlet_centroid'], NS_parameters['inlet_area'])
    # Update this in NS parameters and expression
    #NS_expressions.update(dict(u_in= inflow_Vprof)) 
    


# ---------------------------- Read mesh and obtain useful info -----------------------------#

# all length scales are based on the mesh (in [meters] for most cases)
def mesh(mesh_name, mesh_path, **NS_namespace):
    # Load the volume mesh .xml
    mesh = Mesh(mesh_path + mesh_name)

    # to read .h5 mesh
    #h5   = HDF5File(mesh.mpi_comm(), mesh_path + mesh_name, "r")
    #h5.read(mesh, "/mesh", False)

    print('Mesh min/max coords [mm]:', mesh.coordinates().min()*1000, mesh.coordinates().max()*1000)

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
    wall_tag, inlet_tag, outlet_tag = 1, 2, 3
    dim = mesh.geometry().dim()
    #x   = SpatialCoordinate(mesh)

    #sub_domains = get_subdomains(mesh, mesh_path, BC_file, **NS_namespace)
    sub_domains = MeshFunction("size_t", mesh, mesh_path+BC_file) 
  

    ### --------------- Sanity checks of boundaries ------------ ###
    # Surface integrand over the boundaries
    ds_inlet  = ds(inlet_tag,  domain=mesh, subdomain_data=sub_domains) # don't comment this line!!
    ds_wall   = ds(wall_tag,   domain=mesh, subdomain_data=sub_domains)
    ds_outlet = ds(outlet_tag, domain=mesh, subdomain_data=sub_domains)

    #center = NS_parameters["inlet_centroid"]
    #area   = NS_parameters["inlet_area"]

    #"""
    wall_area   = assemble(Constant(1.0)*ds_wall)
    inlet_area  = assemble(Constant(1.0)*ds_inlet)
    outlet_area = assemble(Constant(1.0)*ds_outlet)

    print("wall_area   [m2] = ", wall_area)
    print("inlet_area  [m2] = ", inlet_area)
    print("outlet_area [m2] = ", outlet_area)
    #"""
    
   
    ### --------------- Create BCs: Velocity ------------ ###

    # Create an array to hold the velocity BC
    bcu =[[],[],[]]


    ### 1. Inlet velocity (Poiseuille parabola)

    # Prescribe poiseuille BC at inlet
    uin = make_poiseuille_velocity(mesh, ds_inlet) #make inlet profile
    #bci = DirichletBC(V, Constant(1.0), sub_domains, inlet_tag)
    bci = [DirichletBC(V, ilt, sub_domains, inlet_tag) for ilt in uin]
    
    for j in range(dim):
        bcu[j].append(bci[j])


    ### 2. Noslip condition at wall  
    noslip = Constant(0.0)
    bcu_wall  = DirichletBC(V, noslip, sub_domains, wall_tag) # this is for each velocity component (and it should hold for all 3 directions)

    # Add no-slip BC at the wall for each velocity component   
    for bcui in bcu:
        bcui.append(bcu_wall)


    #bcu = [[],[],[]]
    #bcu[0] = [bcu_wall, bcu_inlet[0]]
    #bcu[1] = [bcu_wall, bcu_inlet[1]]
    #bcu[2] = [bcu_wall, bcu_inlet[2]]
    
    ### --------------- Create BCs: Pressure ------------ ###
    #Pressure BCs: Dirichlet for outlet pressure -> sets a reference pressure
    outflow = Constant(0.0) #Expression("p", p = 0, degree=1)
    bcp_outlet = DirichletBC(Q, outflow, sub_domains, outlet_tag) 

    
    ### --------------- Sanity checks of the BC ------------ ###
    # 1. Check type of BCs
    #bcs_dict = dict(u0= bcu[0], u1= bcu[1],u2= bcu[2], p = [bcp_outlet])
    #for key, bc_list in bcs_dict.items():
    #    print(key, [type(bc) for bc in (bc_list if isinstance(bc_list, (list,tuple)) else [bc_list])])

    # 2. Check max velocity (magnitude of velocity at center = umax)
    #u_center = np.array([u_expr(center) for u_expr in uin])
    #print("Centerline velocity = ", u_center)

    # 3. Check that all coefficients are finite
    #for name, expr in zip(("u_x","u_y","u_z"), uin):
    #    tmp = Function(V); tmp.interpolate(expr)
    #    m, M = tmp.vector().min(), tmp.vector().max()
    #    print(name, m, M, np.isfinite(m), np.isfinite(M))



    print("Finished creating BCs ...")

    return dict(u0= bcu[0],         #x-component of velocity 
                u1= bcu[1],         #y-component of velocity 
                u2= bcu[2],         #z-component of velocity 
                p = [bcp_outlet])   #Prescribed inlet, zero outlet
    

def make_poiseuille_velocity(mesh, ds_inlet, **NS_namespace):

    dim = 3
    center = NS_parameters["inlet_centroid"]
    area   = NS_parameters["inlet_area"]

    # ----------------------- Obtain inlet parameters ---------------------- #
    
    ### 1. Compute the area-weighted average normal ###

    # Obtain raw normals
    # n_raw[i]: i-th component (i = 0,1,2) of the unit outward normal vector on each boundary facet
    n_raw = FacetNormal(mesh)

    # Average the normals over the inlet
    n_avg  = np.array([assemble(n_raw[i]*ds_inlet) for i in range(dim)])
    
    # Calculate the length of average normal components (~ inlet area) -> used for normalization
    n_len  = np.sqrt(sum([n_avg[i]**2 for i in range(dim)]))

    # Normalize average normals -> normal: unit vector representing the average outward normal of the inlet patch
    normal  = n_avg/n_len


    ### 2. Compute other parameters
    n0, n1, n2 = normal[0], normal[1], normal[2]
    c0, c1, c2 = center[0], center[1], center[2]
    R = np.sqrt(area/np.pi)
    Qin    = float(1.43 * (2*R)**2.55)
    u_max  = 2.0 * Qin / area   
    Reynolds = u_max*2*R/NS_parameters["nu"]
    max_dt = 0.5*mesh.hmin()/u_max  # based on CFL = 0.5

    print(f"Minimum mesh size [mm] = {mesh.hmin()*1000:.4f}")
    print(f"Suggested timestep [s] = {max_dt:0.6f}")

    print (f"Inlet properties: "
           f"R [mm] = {R*1000:.4f}, "
           f"Q [ml/s] = {Qin*1e6:.4f}, "
           f"umax [m/s] = {u_max:.4f}, "
           f"Reynolds = {Reynolds:.1f}, "
           #f"centroid = {center}, "
           f"normal = {normal}")


    # -------------------- Create Expressions for each direction ------------------- #
    # Obtain inlet poiseuille velocity (one component per axis)
    uin_expressions = [[],[],[]]

    """
    for comp in range(3):           
        args = {
            "center":           center,
            "area":             area,
            "normal":           normal,
            "normal_component": normal[comp]}

        u_expr = Poiseuille(center=center, area=area, normal=normal, normal_component = normal[comp], degree = 2)
        #u_expr = Poiseuille(args, degree = 2)
        #u_expr.init({"center": center, "area": area, "normal": normal, "normal_component": normal[comp]})
        uin_expressions[comp] = u_expr
    """

    kernel = (
        "-ncomp * umax * (1.0 - "
        "( pow((x[0]-c0) - n0 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
        "  pow((x[1]-c1) - n1 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) + "
        "  pow((x[2]-c2) - n2 * ((x[0]-c0)*n0 + (x[1]-c1)*n1 + (x[2]-c2)*n2), 2) ) "
        " / (R*R) )"
    )

    for j in range(dim):      
        #uin_expressions[0] = Expression("-ncomp * umax * (1 - (pow(x[0]-c0,2) + pow(x[1]-c1,2) + pow(x[2]-c2,2))/(R*R) )", ncomp = normal[0], umax=u_max, c0 = c0, c1=c1, c2=c2, R= R, degree = 2)
        uin_expressions[j] = Expression(kernel, ncomp = normal[j], umax=u_max,
                                        c0=c0, c1=c1, c2=c2,
                                        n0=n0, n1=n1, n2=n2,
                                        R=R, degree=2)

    return uin_expressions

"""
class Poiseuille(UserExpression):
    
    #Component u_i = -u_max (1 - (r/R)^2) * n_i.
    
    #* `R`                – radius               [m] (float)
    #* `center`           – centre of inlet      [m] (Point)
    #* `normal`           – outward normal       (Point, unit length)
    #* `normal_component` – normal component     (nx,ny,nz)
    #* `u_max`            – centre-line velocity [m/s] (scalar Constant)
    
    def __init__(self, *, center, area, normal, normal_component, degree=2, **kwargs):
        super().__init__(degree=degree, **kwargs)

        self.center = Point(center) #Point(*args["center"])
        self.area   = float(area)
        self.R      = np.sqrt(self.area/np.pi)
        self.Qin    = float(1.43 * (2*self.R)**2.55) #float(args.get("Q_in", 1.43 * (2*self.R)**2.55))   # total inlet flow rate [m3/s]
        self.u_max  = 2.0 * self.Qin / self.area                         # max inlet velocity [m/s] (based on Poiseuille flow)

        self.normal = Point(*normal)           # (nx,ny,nz)
        self.n_comp = float(normal_component)
        
        self.c0, self.c1, self.c2 = self.center
        self.n0, self.n1, self.n2 = self.normal

        #print('Q_in [ml/s] , u_max [m/s] =', self.Qin*1e6, self.u_max)


    def eval(self, value, x):
  
        # distance of point from center of inlet
        dist_x = x[0] - self.c0
        dist_y = x[1] - self.c1
        dist_z = x[2] - self.c2

        # Project distance onto normal -> divide dist vector to normal and in-plane components: dist = dist_inlet + dist_n
        dist_n_len = dist_x*self.n0 + dist_y*self.n1 + dist_z*self.n2 # length of normal component of dist
        
        dist_n_x = dist_n_len*self.n0
        dist_n_y = dist_n_len*self.n1
        dist_n_z = dist_n_len*self.n2

        # In-plane distance squared
        r2  = (dist_x - dist_n_x)**2 + (dist_y - dist_n_y)**2 + (dist_z - dist_n_z)**2

        # parabolic profile
        profile  = self.u_max * (1.0 - r2 / (self.R)**2)

        # Value of inlet velocity in the specified direction 
        value[0] = -self.n_comp * profile   # minus makes inflow  
        
        #print('u_comp = ', value[0])
        #print('Poiseuille is evaluated ...')

    def value_shape(self):
        return ()
"""


"""
def get_inflow_Vprofile(center, area, **NS_namespace):
    
    # Get area and centroid of the inlet
    #center, area = get_inlet_params(mesh, sub_domains) #[-21.6647, 16.2042, -29.5128]

    #center = NS_parameters["inlet_centroid"]
    #area = NS_parameters["inlet_area"]

    diameter = np.sqrt(4*area/np.pi)
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
"""


# ---------------------------- Oasis Functions -----------------------------#
# only called once, before the first time‐step
def pre_solve_hook(mesh, u_, p_, AssignedVectorFunction, **NS_namespace):
    
    uv = AssignedVectorFunction(u_, "Velocity")
    #normals = FacetNormal(mesh)

    #return dict(uv= AssignedVectorFunction(u_, "Velocity"), n= FacetNormal(mesh))
    return dict(uv=uv) #, normals=normals)



def start_timestep_hook(t, **NS_namespace): #originally it was P_in instead of u_in
   #u_in.t = t
   #P_in.t = t
   pass
    

def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass


# Called at each time-step
def temporal_hook(u_, p_, mesh, tstep, uv, print_intermediate_info, plot_interval, **NS_namespace):
    
    # I/O
    if tstep % print_intermediate_info == 0:
        #print("Continuity ", assemble(dot(u_, n) * ds()))
        pinlet = p_.get_local()
        print("pressure gradient: ", pinlet)

    if tstep % plot_interval == 0:
        max_u = max(u_[0].vector().get_local().max(), u_[1].vector().get_local().max())
        #print("tstep= ", tstep, "and umax=", u_max)
        print("tstep= ", tstep, ": worst CFL", NS_parameters['dt']*max_u/mesh.hmin())



def theend_hook(uv, p_, **NS_namespace):
    #uv()
    pass



