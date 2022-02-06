# PyWBGT

This repository contains Cython source code for estimating wet bulb globe temperature (WBGT) from datasets of standard meterological measurements using models developed by Liljegren et al (2008) [1].  

****
### What is WBGT?
WBGT is a widely-applied heat stress index, enjoying the advantages of a simple physical interpretation, covering all four ambient factors (temperature, humidity, wind and radiation) contributing to heat stress, and having well established safety thresholds to guide activity modification within the military, occupational and athletic settings. It is constructed as a linear combination of natural wet bulb temperature (Tw), black globe temperature (Tg) and dry bulb temperature (Ta): WBGT=0.7Tw+0.2Tg+0.1Ta [2].

Several models exist for estimating WBGT from meteorological data among which the model developed by Liljegren is recommended [3]. Liljegren's model was written in C and FORTRAN language. We rewrote it in Cython which is fast, easy to use in Python and scales well for large dataset such as climate model output.

****
### What is in this repository?
- `./src/`: Cython source files (```.pyx``` file) for calculating WBGT and cosine zenith angle (needed for WBGT calculation); Cython source file needs to be compiled first generating shared object files (```.so``` file) that can be directly imported in Python. Setup tools (```.py``` file) are recommended for build Cython source file. This directory already contains the compiled shared object files, although users can also compile themselves.
- `./Jupyter_notebooks/`: A jupyter nobtebook introducing the usage of our code.
- `environment.yml` a YAML file that can be used to build conda environment containing all needed python packages.

****
### Future plans
We plan to build and distribute a Python package for WBGT calculation.

****
### How to use the Jupyte notebooks
Before using the Jupyter notebook, users need to install dependent Python packages listed in `environment.yml`. Users should be able to import our WBGT module in Python implementation if they put `.so` file under their directory. If that does not work, users can compile Cython source files themselves with the following command:
- for Intel compiler: 
  - `LDSHARED="icc -shared" CC=icc python setupWBGT.py develop`; 
  - `LDSHARED="icc -shared" CC=icc python setupcoszenith.py develop`
- for gcc compiler: 
  - `python setupWBGT.py build_ext --inplace`; 
  - `python setupcoszenith.py build_ext --inplace`
  
For introduction to Cython, please refer to https://cython.readthedocs.io/en/latest/

****
### Citation
If you want to use our code, please consider cite `Kong, Qinqin, and Matthew Huber. “Explicit Calculations of Wet Bulb Globe Temperature Compared with Approximations and Why It Matters for Labor Productivity.” Earth’s Future, January 31, 2022. https://doi.org/10.1029/2021EF002334.`


****
### References

[1] Liljegren JC, Carhart RA, Lawday P, Tschopp S, Sharp R. Modeling the wet bulb globe temperature using standard meteorological measurements. J Occup Environ Hyg. 2008;5(10):645-55. 

[2] Yaglou CP, Minard D. Control of heat casualties at military training centers. AMA Arch Ind Health. 1957;16(4):302-16. 

[3] Lemke B, Kjellstrom T. Calculating workplace WBGT from meteorological data: a tool for climate change assessment. Ind Health. 2012;50(4):267-78. 


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
