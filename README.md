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


## ToDo : 

### Fitzugh & Nagumo Model

*Coming Soon*

### Hindmarsh & Rose Model

*Coming Soon*
