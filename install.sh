#!/bin/bash

#conda update --all
# pip install emcee
# pip install batman-package
# pip install ldtk
# pip install pandas
# pip install pypdf2
# pip install corner
# pip install pillow
# pip install pyyaml

python setup.py develop --user

# mkdir -p ../exotep_analysis
# mkdir -p ../exotep_results
# mkdir -p ../../DataSpitzer/ExtractedPhotometry
# mkdir -p ../../DataHST

# #mkdir -p ../../DataHST
# #mkdir -p ../../Kepler
# #cp tests/exotep_analysis/run_ExoTEP.py   ../exotep_analysis/
# cp tests/DataSpitzer/ExtractedPhotometry/WASP_107_b_62712320ch2.dat   ../../DataSpitzer/ExtractedPhotometry/
#cp tests/AllPla.pickle   ../


