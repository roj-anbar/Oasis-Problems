"""
Created on Aug 12 2024
Modified by Rojin Anbarafshan from original code written by Anna Haley
Contact: rojin.anbar@gmail.com

Incompressible Navier-Stokes equations for channel flow with pulsatile inlet pressure (Womersley flow)
on the unit square using the Oasis solver.
"""

from __future__ import print_function
from oasis import *
from oasis.problems import *
from oasis.problems.NSfracStep import *
from oasis.common import * #added to be able to use io.py functions
from dolfin import *
import numpy as np 
import sys

def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):
    Re = 1 #300 
    NS_parameters.update(
        alpha = 1,
        H = 1, #channel width (dimonsionless)
        period=1,
        cycles=2,
        dt=0.01,
        Re=Re,
        mesh_location= '/scratch/s/steinman/ranbar/2D/Womersley/',
        mesh_name= 'mesh.xml',
        save_step=1, #10,
        folder='results/',
		linear_solver="mumps",
        checkpoint=100,
        plot_interval=100,
        print_intermediate_info = 1e7,
        print_velocity_pressure_convergence=True)
    NS_parameters.update(T = NS_parameters['cycles']*NS_parameters['period'])
    NS_parameters.update(omega = 2*pi/NS_parameters['period'])
    NS_parameters.update(nu = (NS_parameters['H']**2) * NS_parameters['omega'] / (4 * (NS_parameters['alpha']**2))) #dimonsionless
        
    

    inflow_Pprof = get_inflow_Pprofile(NS_parameters['period'])
    #average_inlet_velocity = get_ave_inlet_velocity(NS_parameters['Re'])
    #inflow_prof = get_inflow_profile(average_inlet_velocity)
    #max_velocity = (3/2)*average_inlet_velocity
    #NS_parameters.update(
    #    ave_inlet_velocity=average_inlet_velocity,
    #    )
    
    #NS_expressions.update(dict(u_in=Expression(inflow_prof, degree=2))) 
    NS_expressions.update(dict(P_in=Expression(inflow_Pprof, t=0., degree=1))) 


def create_bcs(V, Q, mesh, mesh_location, **NS_namespace):
    sub_domains = get_sub_domains(mesh_location, mesh)

    bcu_walls = DirichletBC(V, Constant(0), sub_domains, 1) #sides velocity (for each component)
    #bcu_inlet = DirichletBC(V, Constant(0), sub_domains, 2) #inlet velocity (for each component)
    #bc1 = DirichletBC(V, NS_parameters['ave_inlet_velocity'], sub_domains, 2) #inlet velocity (constant profile)
    #bc2 = DirichletBC(V, NS_expressions['u_in'], sub_domains, 2) #inlet velocity (parabolic profile)
    bcp_inlet = DirichletBC(Q, NS_expressions['P_in'], sub_domains, 2) #inlet pressure
    bcp_outlet = DirichletBC(Q, 0, sub_domains, 3) #outlet pressure

    return dict(u0=[bcu_walls], #0 on sides and inlet
                u1=[bcu_walls], #0 on sides and inlet
                p=[bcp_inlet, bcp_outlet]) #Prescribed inlet, zero outlet


def pre_solve_hook(mesh, u_, AssignedVectorFunction, **NS_namespace):
    return dict(uv= AssignedVectorFunction(u_, "Velocity"), n= FacetNormal(mesh))


"""
# for debug purposes
# for checking the first timestep (use above function instead for actual sims)
def pre_solve_hook(mesh, V, Q, newfolder, folder, tstepfiles, p_, u_, u_components,AssignedVectorFunction, **NS_namespace):
    uv= AssignedVectorFunction(u_, "Velocity")
    #u_comp = map(lambda x: "u"+str(x), range(dim))# velocity components
    #to save the solution at a specific timestep
    save_tstep_solution_h5(0, p_, u_, folder,  tstepfiles,constrained_domain=None,output_timeseries_as_vector=False, # type: ignore
                           AssignedVectorFunction=uv,u_components=u_components,scalar_components=None, NS_parameters=NS_parameters) 
    return dict(uv=uv, n= FacetNormal(mesh))
"""

def start_timestep_hook(t, P_in, **NS_namespace):
    P_in.t = t
    
def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass

def temporal_hook(u_, p_, mesh, tstep, print_intermediate_info,
                  uv, n, plot_interval, **NS_namespace):
    if tstep % print_intermediate_info == 0:
        print("Continuity ", assemble(dot(u_, n) * ds()))

    #if tstep % plot_interval == 0:
        #u_max = u_[0].vector().get_local().max() #max(u_[0].vector().get_local().max(), u_[1].vector().get_local().max())
        #print("worst possible Courant number", NS_parameters['dt']*u_max/mesh.hmin())


def theend_hook(uv, p_, **NS_namespace):
    uv()
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')



# all length scales are unitless
class Noslip(SubDomain): #other way is to mark whole boundary, inflow and outflow will overwrite later
    def inside(self, x, on_boundary):
        return on_boundary and not (near(x[0], -0.5, 1E-6) or near(x[0], 0.5, 1E-6))

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -0.5, 1E-6) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.5, 1E-6) and on_boundary #second argument was originally 0.24

'''
def calc_parabola_vertex(inlet_velocity):
    x1, x2 = 0, 0.008
    y1, y2 = 0, 0
    x3 = np.mean([x1,x2])
    y3 = (3/2)*inlet_velocity

    denom = (x1-x2) * (x1-x3) * (x2-x3);
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;

    return A,B,C
'''

def mesh(mesh_name, mesh_location, **NS_namespace):
    m = Mesh(mesh_location + mesh_name) #Mesh(mesh_location + "stenosis.xml")
    return m


def get_sub_domains(mesh_location, mesh, **NS_namespace):
    sub_d = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_d.set_all(0)

    wall = Noslip()
    wall.mark(sub_d, 1)

    inlet = Inlet()
    inlet.mark(sub_d, 2)

    outlet = Outlet()
    outlet.mark(sub_d, 3)

    return sub_d

'''
def get_ave_inlet_velocity(Re, **NS_namespace):
    inlet_width = 0.02 #[m] #0.008
    average_inlet_velocity = Re*NS_parameters['nu']/inlet_width
    print('Inlet Velocity:', average_inlet_velocity)
    return average_inlet_velocity
'''

'''
def get_inflow_profile(ave_inlet_velocity, **NS_namespace):
    A, B, C = calc_parabola_vertex(ave_inlet_velocity)

    inflow_prof = '{}*x[1]*x[1] + {}*x[1] + {}'.format(A, B, C)
    return inflow_prof
'''

def	get_inflow_Pprofile(period, **NS_namespace):
    rho = 1 #dimonsionless
    K = 1   
    inflow_Pprof = '{}*{}*sin(2*pi*t/{})'.format(rho,K,period)
	#inflow_Pprof = '{}*{}*(0.75+1.5*sin(2*pi*t/{}))'.format(rho,max_vel*max_vel,period)
    #print('Inflow Pressure:', inflow_Pprof)
    return inflow_Pprof
