# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:11:39 2020

@author: KugelBlitZZZ

Hindmarsh-Rose model of the neuron

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


class Neuron():
    
    
    def __init__(self , r , s , xr , a , b , c, d, I) : 
        self.r = r
        self.s, self.xr, self.a, self.b  = s , xr, a, b  
        self.c, self.d, self.I = c, d, I


    def hind_rose(self) :
        """Hindmarsh-Rose system of equations (x,y,z)"""
        def pre_hind_rose(t, vars) : 
        
            x, y, z = vars
            return np.array([y - self.a*x**3 + self.b*x**2 - z + self.I,\
                             self.c - self.d*x**2 - y,\
                             self.r *(self.s*(x - self.xr) - z)])
        return pre_hind_rose
     
        
    def simulate_hind_rose(self, x_init, y_init, z_init, Tmax):
        """Method to simulate the model based on given initial conditions.
        
        Parameters
        ----------
        x_init : int
            The initial x value of the model.
        y_init : int
            The initial y value of the model.
        z_init : int
            The initial z value of the model.
        Tmax : int
            maximum time in milliseconds for the solver.

        Returns
        -------
        None. but saves a .mp4 video of the simulation
        """

    
        # First off, let's solve the system from our initial conditions
        sol = solve_ivp(self.hind_rose(),
                        [0, Tmax],
                        (x_init, y_init, z_init),
                        t_eval=np.linspace(0, Tmax, 500))
    
        # Let's separate the solutions and the time vector
        tt, tx, ty, tz = sol.t, sol.y[0], sol.y[1], sol.y[2]
    
        # And now let's create the animation
        fig = plt.figure(figsize=(12, 5), dpi=150)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
        l2d, = ax1.plot(tt, tx, 'r')
        ax1.set_xlabel('t')
        ax1.set_ylabel('$x$', rotation=0)
        l3d, = ax2.plot(tx, ty, tz, 'b')
        ax2.set_zlabel('$z$')
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        ax2.set_zlim(1.6,2.2)
        fig.suptitle(
            "Hindmarsh and Rose model Simulation \n Initial conditions : $x$ = %s, $y$ = %s, $z$ = %s"
            % (x_init, y_init, z_init))
        
        def animate(i) :
            l2d.set_data(tt[:i], tx[:i])
            data = np.array([tx,ty,tz]) # There is no set_data() for 3d plots
            l3d.set_data(data[0:2, :i])
            l3d.set_3d_properties(data[2, :i])
            return l2d, l3d
        
        anim = FuncAnimation(fig, animate, frames=len(tt), interval = 50)
        anim.save('HRsimulation.mp4', dpi = 150, fps= 30)
  
    

