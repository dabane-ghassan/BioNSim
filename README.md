# biological_neurons
A collection of Python scripts to implement biological models of neurons using Python. 
Numpy, matplotlib, scipy as well as celluloid package were used to animate the output. 

## Hudgkin & Huxley Model

### Input
- For each ion, G is the conductance and Eq is the equilibrium potential.
```python
import hudgkin_huxley as hh

cell = hh.Neuron(voltage=np.linspace(-150, 150, 100), 
                 sodium={'G' : 120., 'Eq' : 120.}, 
                 potassium={'G' : 36., 'Eq' : -12.},
                 leak={'G' : 0.3,'Eq' :10.6})

cell.simulate(V_init=-10, n_init=0, m_init=0, h_init=1, Tmax=50, inj = 15)
``` 
### Output
![HH simulation](Demo/HHsimulation.gif)

## Hindmarsh & Rose Model
- Model parameters should be specified when instantiating the object, and the initial conditions can be changed when simulating the model.
```python
import hindmarsh_rose as hr

cell = hr.Neuron(r = 0.001, s = 4, xr = -8/5, a = 1, b = 3, c = 1, d = 5, I = 2)

cell.simulate_hind_rose(x_init=-1.5, y_init=-10, z_init=2, Tmax=1000)
``` 

![HR simulation](Demo/HRsimulation.gif)

## ToDo : 

### Fitzugh & Nagumo Model

*Coming Soon*

### IKir Model

*Coming Soon*
