# -*- coding: utf-8 -*-
"""
@author: Ghassan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp  


class Neuron():
    """Hudgkin & Huxley Model of the excitable cell"""
    def __init__(self, voltage, sodium, potassium, leak):
        """This instantiates a Neuron object with the following properties
        sodium, potassium and leak should be dictionaries with their
        conductance and equilibrium potential as elements.
        """
        self.voltage = voltage
        self.Gna, self.Ena = sodium['G'], sodium['Eq']
        self.Gk, self.Ek = potassium['G'], potassium['Eq']
        self.Gl, self.El = leak['G'], leak['Eq']

    def __str__(self):
        """This lets the Neuron Object be used by the print() function"""
        return "Hello ! I'm Hudgkin & Huxley Model"

    def an(self, volt):
        """alpha_n in function of voltage, potassium channel constant"""
        return 0.01 * (10 - volt) / (np.exp((10. - volt) / 10.) - 1.)

    def bn(self, volt):
        """beta_n in function of voltage, potassium channel constant"""
        return 0.125 * np.exp(-volt / 80.)

    def am(self, volt):
        """alpha_m in function of voltage, sodium channel constant"""
        return 0.1 * (25. - volt) / (np.exp((25. - volt) / 10.) - 1.)

    def bm(self, volt):
        """beta_m in function of voltage, sodium channel constant"""
        return 4. * np.exp(-volt / 18.)

    def ah(self, volt):
        """alpha_h in function of voltage, sodium channel constant"""
        return 0.07 * np.exp(-volt / 20.)

    def bh(self, volt):
        """beta_h in function of voltage, sodium channel constant"""
        return 1. / (np.exp((30. - volt) / 10.) + 1.)

    def tau(self, alpha, beta):
        """this method calculates the time constant tau 
        for a certain ion channel.
        """
        return 1. / (alpha + beta)

    def finfty(self, alpha, beta):
        """this method calculates the stationary constant
        f_infty for a certain ion channel.
        """
        return alpha / (alpha + beta)

    def INa(self):
        """this calculates the sodium transient current using previous
        class methods.
        """
        m = self.finfty(self.am(self.voltage), self.bm(self.voltage))
        h = self.finfty(self.ah(self.voltage), self.bh(self.voltage))

        return self.Gna * m**3. * h * (
            self.voltage - self.Ena
        )  # Ohm's law for ion channels I = G*(V - Eeq)

    def IK(self):
        """this calculates the potassium persistant current using previous
        class methods.
        """
        n = self.finfty(self.an(self.voltage), self.bn(self.voltage))

        return self.Gk * n**4. * (
            self.voltage - self.Ek
        )  # Ohm's law for ion channels I = G*(V - Eeq)

    def IL(self):
        """this calculates the leak channel current."""
        return self.Gl * (self.voltage - self.El)

    def time_constants(self):
        """This method creates an animation for the time constants
        of our model for a given array of voltages.
        """
        tn = self.tau(self.an(self.voltage), self.bn(self.voltage))
        tm = self.tau(self.am(self.voltage), self.bm(self.voltage))
        th = self.tau(self.ah(self.voltage), self.bh(self.voltage))
        fig, ax = plt.subplots(dpi=150)
        t1, t2, t3 = plt.plot(self.voltage, tm, 'r',
                              self.voltage, th, 'b',
                              self.voltage, tn, 'y',
                              linestyle='solid')
        plt.legend([t1, t2, t3], [r'$\tau_m$', r'$\tau_h$', r'$\tau_n$'])
        plt.suptitle("Time constants for ion channel kinetics")
        plt.xlabel("Voltage in mV")

    def stationary_constants(self):
        """This method creates an animation for the stationary constants
        of our model for a given array of voltages.
        """
        fn = self.finfty(self.an(self.voltage), self.bn(self.voltage))
        fm = self.finfty(self.am(self.voltage), self.bm(self.voltage))
        fh = self.finfty(self.ah(self.voltage), self.bh(self.voltage))
        fig, ax = plt.subplots(dpi=150)
        f1, f2, f3 = plt.plot(self.voltage, fm, 'r',
                              self.voltage, fh, 'b',
                              self.voltage, fn, 'y')
        plt.legend([f1, f2, f3],
                   [r'$m\infty$', r'$h\infty$', r'$n\infty$'])
        plt.suptitle("Stationary constants for ion channel kinetics")
        plt.xlabel("Voltage in mV")

    def show_currents(self):
        """This method creates an animation for the 3 currents in our model
        calculated based on a given array of voltages.
        """

        fig, ax = plt.subplots(dpi=150)
        i1, i2, i3 = plt.plot(self.voltage, self.IK(), 'r',
                              self.voltage, self.INa(), 'g',
                              self.voltage, self.IL(), 'b')
        plt.legend([i1, i2, i3], [r'$I_{K^+}$', r'$I_{Na^+}$', r'$I_{L}$'])

        plt.ylim(-200, 500)
        plt.xlabel("Voltage in mV")
        plt.ylabel('Current in mA')
        plt.suptitle("$I-V$ curves in Hudgkin and Huxley Model")

    def simulate(self, V_init, n_init, m_init, h_init, Tmax, inj=0):
        """This method simulates the H&H model starting from initial conditions
        given as parameters for a given period of time Tmax, it uses
        scipy.integrate.solve_ivp() function to solve the system of four
        differential equations.
        """
        def hudgkin_huxley(t, vars):
            """The four nonlinear differential equations should be wrapped
            inside a function.
            """
            V, n, m, h = vars
            return [
                -self.Gna * m**3 * h * (V - self.Ena) - self.Gk * n**4 *
                (V - self.Ek) - self.Gl * (V - self.El) + inj,
                (self.an(V) * (1 - n)) - (self.bn(V) * n),
                (self.am(V) * (1 - m)) - (self.bm(V) * m),
                (self.ah(V) * (1 - h)) - (self.bh(V) * h)
            ]

        solution = solve_ivp(
            hudgkin_huxley,
            [0, Tmax],  # Solve the system 
            (V_init, n_init, m_init, h_init),  # Initial conditions as a tuple
            t_eval=np.linspace(0, Tmax, 250))

        tt, vt, nt, mt, ht = solution.t, solution.y[0], solution.y[
            1], solution.y[2], solution.y[3]  # Seperate the solutions

        # Now let's calculate currents and conductances based on the solutions
        GNA, INA = self.Gna * mt**3 * ht, self.Gna * mt**3 * ht * (vt -
                                                                   self.Ena)
        GK, IK = self.Gk * nt**4, self.Gk * nt**4 * (vt - self.Ek)
        IL = self.Gl * (vt - self.El)

        # Now let's start the plotting and the animation process
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=150)

        l1, = axes[0].plot(tt, vt, 'g', linestyle='solid')
        axes[0].legend([l1], [r'$V(t)$'])  # V(t) in function of time

        i1, i2, i3 = axes[1].plot(tt, INA, 'y',
                                  tt, IK, 'r',
                                  tt, IL, 'b',
                                  linestyle='solid') # Currents in fucntion of time
        axes[1].set_xlabel('Time')
        axes[1].legend([i1, i2, i3],
                       [r'$I_{Na^+}$', r'$I_{K^+}$', r'$I_{L}$'])
        axes[1].set_title("""Hudgkin & Huxley Model Simulation,
            Initial conditions : $V$ = %s, $n$ = %s, $m$ = %s, $h$ = %s, $T_{max}$ = %s, $I_{injected}$ = %s"""
                          % (V_init, n_init, m_init, h_init, Tmax, inj))

        g1, g2 = axes[2].plot(
            tt, GNA, 'y', tt, GK, 'r',
            linestyle='solid')  # Conductances in function of time
        axes[2].legend([g1, g2], [r'$g_{Na^+}$', r'$g_{K^+}$'])
        
        def animate(i) :
            l1.set_data(tt[:i], vt[:i])
            i1.set_data(tt[:i], INA[:i])
            i2.set_data(tt[:i], IK[:i])
            i3.set_data(tt[:i], IL[:i])
            g1.set_data(tt[:i], GNA[:i])
            g2.set_data(tt[:i], GK[:i])
            return l1, i1, i2, i3, g1, g2
        
        anim = FuncAnimation(fig, animate, frames=len(tt), interval = 50)
        anim.save('HRsimulation.mp4', dpi = 150, fps= 30)
        
           
