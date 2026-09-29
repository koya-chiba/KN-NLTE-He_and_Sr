# reqlocal
This is rate equation solver for helium (He) in supernova and kilonova ejecta.

To explore the physical conditions of the 1 um feature 
in the early-phase spectra of AT2017gfo, 
simple rate equation solver for strontium (Sr)
which just solve the ionization balance is implemented. 

## File structure 
  - data: data directory
  - base.py: defines solver parameters 
  - constants.py: defines physical constants  
  - environment.py: defines ejecta parameters
  - ion.py: defines the class which solve the ionization balance
  - req.py: defines the main body of the solver 
  - transition.py: makes rate matrix
