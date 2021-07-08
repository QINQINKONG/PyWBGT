# PyWBGT

This repository contains Cython source code that can be used to estimate wet bulb globe temperature (WBGT) from datasets of standard meterological measurements using models developed by Liljegren et al (2008) [1].  

****
### What is WBGT?
WBGT is a widely-applied heat stress index, enjoying the advantages of a simple physical interpretation, covering all four ambient factors (temperature, humidity, wind and radiation) contributing to heat stress, and having well established safety thresholds to guide activity modification within the military, occupational and athletic settings. It is constructed as a linear combination of natural wet bulb temperature (Tw), black globe temperature (Tg) and dry bulb temperature (Ta): WBGT=0.7Tw+0.2Tg+0.1Ta [2].

Several models exist for estimating WBGT from meteorological data among which the model developed by Liljegren is recommended [3]. Liljegren's model was written in C and FORTRAN language. We rewrote it in Cython which is fast, easy to use in Python and scales well for large dataset such as climate model output.
****
### What is in this repository?
- ```./src/```: Cython source file for calculating WBGT and cosine zenith angle (needed for WBGT calculation)
- ```./Jupyter_notebooks/```: A jupyter nobtebook introducing the usage of our code and several other jupyter notebooks and processed dataset that can be used to reproduce figures in "Explicit Calculations of Wet Bulb Globe Temperature compared with approximations and why it matters for labor productivity" (submitted to Geohealth) by Qinqin Kong and Matthew Huber. 

### Future plans
We plan to build and distribute a Python package for heat stress metrics calculation. It will not only include the WBGT code here but also code for several other heat stress metrics like thermodynamic wet bulb temperature.

### How to use the Jupyte notebooks
By launching the Binder projected created for this repository, users will be able to run the Jupyter notebooks without installing any package by thenselves. 
If users want to run the notebooks or use our code in their personal environment, they can either place the ```.so``` shared object file under their personal directory, or compile the code using setup tools (to get the shared object file) by themselves. The following command can be used for compiling Cython source file:
- for Intel compiler: ```LDSHARED="icc -shared" CC=icc python setupWBGT.py develop```; ``LDSHARED="icc -shared" CC=icc python setupcoszenith.py develop```
- for gcc compiler: ```python setupWBGT.py build_ext --inplace```; ```python setupcoszenith.py build_ext --inplace```

### References

[1] Liljegren JC, Carhart RA, Lawday P, Tschopp S, Sharp R. Modeling the wet bulb globe temperature using standard meteorological measurements. J Occup Environ Hyg. 2008;5(10):645-55. 
[2] Yaglou CP, Minard D. Control of heat casualties at military training centers. AMA Arch Ind Health. 1957;16(4):302-16. 
[3] Lemke B, Kjellstrom T. Calculating workplace WBGT from meteorological data: a tool for climate change assessment. Ind Health. 2012;50(4):267-78. 
