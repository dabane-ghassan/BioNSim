# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:10:47 2020

@author: Ghassan Dabane

Fitzugh-Nagumo model of the biological neuron

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import bisect
from scipy.integrate import solve_ivp


class Neuron() : 
    def __init__(self, a, b, tau) : 
        self.a, self.b, self.tau = a, b, tau 
        self.eq_equation = lambda v, I: v - (v**3 / 3) + I - ((v + a) / b)


    def eq_coordinates(self, I):
        """This function sends back the equilibrium coordinates of The 
        Fitzugh-Nagumo Model for the equilibrium equation specified as 
        the eq_equation.
        """
        vstar = bisect(
            self.eq_equation, -2, +2, I
        )  #solve eq_equation between -2 and +2 with bisect function from scipy
        wstar = (vstar + self.a) / self.b
        return vstar, wstar
    
    
    def vnull(self, v, I):
        "V-nullcline of the Fitzugh-Nagumo model"
        return v - v**3 / 3 + I
    
    
    def wnull(self, v):
        "w-nullcline of the Fitzugh-Nagumo model"
        return (v + self.a) / self.b
    
    
    def vdot(self, v, w, I):
        """this function sends back the values of dvdt"""
        return v - v**3 / 3 - w + I
    
    
    def wdot(self, v, w):
        """this function sends back the values of dwdt"""
        return (v + self.a - self.b * w) / self.tau
    
    
    def fitz_nagu(self, t, z, I):
        """This function contains the equations of the model,
        it will be used with scipy's solve_ivp function in order to solve 
        the system numerically starting from initial conditions.
        """
        v, w = z
        return np.array([self.vdot(v, w, I), self.wdot(v, w)])




    def simulate_fitz_nagu(self, V_init, w_init, I_init, Tmax):
        
        # Determining equilibrium 
        v_star, w_star = self.eq_coordinates(I=I_init)
    
        # Solve the system
        sol = solve_ivp(
            lambda t, z: self.fitz_nagu(t, z, I=I_init), [0, Tmax],
            (V_init, w_init
             ), t_eval=np.linspace(0, Tmax, 150))  # An anonymous function with 
        # fitz-nagu was used because solve_ip
        # doesn't suppport specifiying other parameters for the function to be
        # solved, this can be a work-around as it lets us specify I
    
        tt, vt, wt = sol.t, sol.y[0], sol.y[
            1]  # Time, voltage and recovery variable
    
        # Voltage and arrows
        volt = np.linspace(-5, 5, 100)  # Voltage array between -5 and +5 mV
        x_arrs, y_arrs = np.meshgrid(np.linspace(-3, +3, 15),
                                     np.linspace(-2, +2, 10))
    
        # Figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        s1, = axes[0].plot(vt, wt, 'r')
        ng1, = axes[0].plot(volt, self.vnull(volt, I=I_init), color="orange")
        ng2, = axes[0].plot(volt, self.wnull(volt), 'b')
        eq, = axes[0].plot(v_star, w_star, 'ko', label="Stable Node")
        axes[0].quiver(x_arrs,y_arrs,self.vdot(x_arrs,y_arrs,I_init),
                       self.wdot(x_arrs,y_arrs), color = 'green')
        axes[0].legend(
            [ng1, ng2, eq, s1],
            ['$V$-nullcline', '$w$-nullcline', 'Equilibrium', 'Solution'],
            loc = "upper right")
        axes[0].set_ylim(-2, +2)
        axes[0].set_xlim(-3, 3)
        axes[0].set_ylabel('$w$', rotation=0)
        axes[0].set_xlabel('$V$')
        axes[0].set_title('Phase portrait')

        vg, = axes[1].plot(tt, vt, color='orange')
        wg, = axes[1].plot(tt, wt, 'b')
        axes[1].legend([vg, wg], ['$V(t)$', '$w(t)$'])
        axes[1].set_xlabel('Time')
        axes[1].set_title('Numerical solution')
        fig.suptitle(
            "Fitzugh and Nagumo model Simulation, Initial conditions : $V$ = %s, $w$ = %s, $I$ = %s"
            % (V_init, w_init, I_init))
        
        def animate(i) :

            s1.set_data(vt[:i], wt[:i])
            vg.set_data(tt[:i], vt[:i])
            wg.set_data(tt[:i], wt[:i])
            return s1, vg, wg 
        
        anim = FuncAnimation(fig, animate, interval = 100)
        anim.save('FNsimulation.mp4', dpi = 150, fps= 30)
 
    



    
    
    



