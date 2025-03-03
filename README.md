This repo includes reproductions of part of the results from 

### Accurate Path Integration in Continuous Attractor Network Models of Grid Cells 
#### Burak & Fiete 2009

as well as complementary investigations on the normative validity of this model based on Sorscher _et al._ work. We perform a directed dimensionality reduction of Burak's model. We train a RNN to integrate velocity input in order to accurately track position. The resulting RNN cells have grid cell-like activity but with a different, square pattern. Varying the parameters allows to observe other types of ratemaps (half-planes, hexagonal grids, ...).

This is our final project for Theoretical Neuroscience class at ENS-PSL. An overview of the work is available in the **evaluate_grid.ipynb** notebook


### Results 

![image](https://github.com/user-attachments/assets/0156df65-054c-4788-b61a-0fc6302b8de5)

![image](https://github.com/user-attachments/assets/a1512c6a-68ea-4ff0-a44e-d84186fd3fe4)

![image](https://github.com/user-attachments/assets/44ca88a5-dbbb-46b0-8131-0d65c2e1b1b0)



### Python files descriptions
- **grid.py** The continuous attractor model code
- **options.py** The options object class for defining grid models
- **analysis.py** Functions for analysing the grid cells model
- **plotting.py** Functions to make all the plots in the notebook
- **solvers.py** ODE solvers
- **tilings.py** Functions to generate angle preference tiling


### Sources
- **Accurate Path Integration in Continuous Attractor Network Models of Grid Cells**
*Burak Y, Fiete IR* (2009) PLOS Computational Biology https://doi.org/10.1371/journal.pcbi.1000291

- **A unified theory for the computational and mechanistic origins of grid cells** 
*Ben Sorscher, Gabriel C. Mel, Samuel A. Ocko, Lisa M. Giocomo, Surya Ganguli* (2023) Neuron  https://doi.org/10.1016/j.neuron.2022.10.003
Authors github >>> https://github.com/ganguli-lab/grid-pattern-formation

