#import WaveCal
#reload(WaveCal)

import WaveCal_orig_Lband as WaveCal
#import WaveCal_stelltell_Kband_CB as WaveCal 
reload(WaveCal)
import matplotlib.pylab as plt
import numpy as np
#import Fringe_class as fringe ## 

### CHOOSE BAND ###
band = 'L'
#band = 'K'
#band = 'M'

n_order_K = 6
n_order_L = 5
n_order_M = 2

wave_min_K = np.array([2.34474, 2.27431, 2.20846, 2.14638, 2.08678, 2.03186])
wave_max_K = np.array([2.38127, 2.30955, 2.24218, 2.17881, 2.11799, 2.06194])

### for K-long
#wave_min_K = np.array([2.38157,2.31,2.24245,2.17894,2.11878,2.0617])
#wave_max_K = np.array([2.41566,2.34284,2.27485,2.20861,2.14639,2.08703])

wave_min_L = np.array([3.40170, 3.25510, 3.11947, 2.99529, 2.88066])
wave_max_L = np.array([3.45520, 3.30559, 3.16903, 3.04284, 2.92635])

wave_min_M = np.array([4.95145, 4.64382])
wave_max_M = np.array([5.03151, 4.71965])

modelfile = '/home/cbuzard/Code/PCA/StellarModels/stellarmodel_HD187123_K_ordersonly.dat'

if band == 'K' : 
    n_order = n_order_K
    wave_min = wave_min_K
    wave_max = wave_max_K
if band == 'L' : 
    n_order = n_order_L
    wave_min = wave_min_L
    wave_max = wave_max_L
if band == 'M' : 
    n_order = n_order_M
    wave_min = wave_min_M
    wave_max = wave_max_M

##### FIT DISPERSION SOLUTIONS #####
ObsFilename=['']*n_order

## ORDER 1
ObsFilename[0] = 'WAVE/20170907/HD187123_20170907_0532_wave1.fits'

## ORDER 2
ObsFilename[1] = 'WAVE/20170907/HD187123_20170907_0532_wave2.fits'

## ORDER 3
ObsFilename[2] = 'WAVE/20170907/HD187123_20170907_0532_wave3.fits'     

## ORDER 4
ObsFilename[3] = 'WAVE/20170907/HD187123_20170907_0532_wave4.fits'

### ORDER 5
ObsFilename[4] = 'WAVE/20170907/HD187123_20170907_0532_wave5.fits'   

### ORDER 6
#ObsFilename[5] = 'WAVE/HD187123_20170907_0532_wave1.fits'

# need vbary and vrad for K band wavelength solutions
# from PyAstronomy import pyasl
# RA = 360*13/24.+47/60.+16.04/3600 # tau boo
# Dec = 17.+27/60.+24.39/3600       # tau boo
# heli, bary = pyasl.baryvel(JD, deq=2000.0)
# vh, vb = pyasl.baryCorr(JD, RA, Dec, deq=2000.0)

vrad = -16.965
vbary = 15.8769330581   # 4/6/2015 13:22
#vbary = 17.0942194754   # 5/8/2015 10:30

#2015apr06
WaterAbundance = 0.48455072339
MethaneAbundance = 1.07139188017
CarbonDioxideAbundance = 1.42491674
CarbonMonoxideAbundance = 0.835038289258 

#apr08 abundances # tboo  2015
#WaterAbundance = 0.48455072339
#MethaneAbundance = 1.07139188017
#CarbonDioxideAbundance = 1.42491674
#CarbonMonoxideAbundance = 0.835038289258 

for order,filename in enumerate(ObsFilename) :
    #if order != 3 : continue
    print '-----------------------------------------------------'
    print '-----------------------------------------------------'
    print 'ORDER', order, 'ORDER', order, 'ORDER', order, 'ORDER', order, 'ORDER', order, 'ORDER', order 
    print '-----------------------------------------------------'
    if band == 'K' : 
        cal = WaveCal.WaveCal_st(ObsFilename=filename, ModelFilename=modelfile,wave_min=wave_min[order], wave_max=wave_max[order], vrad=vrad, vbary=vbary,
                             H2O_scale=WaterAbundance, CH4_scale=MethaneAbundance, CO2_scale=CarbonDioxideAbundance, 
                             CO_scale = CarbonMonoxideAbundance)
    if band == 'L' : 
        cal = WaveCal.WaveCal(ObsFilename=filename, wave_min=wave_min[order], wave_max=wave_max[order])

print cal

##### QUICK REDUCTION TO GET 1D WAVE SPECTRA FOR FITTING #####

#path = '/home/gablakers/NirspecData/2015nov21/spec/'
#Flat_KL = [(239,366)] 
#FlatDark_KL = [(129,133)]  
#ObsDark = [(230,232)] #Darks to not match observations
#
### 51 Peg
##SciRanges = [(354,357)] #397)]
##StdRanges = [(398,401)]
#
### HD 88133
#SciRanges = [(142,145)]
#StdRanges = [(138,141)]
#
#qr = WaveCal.QuickRed(flat_range=Flat_KL, flat_dark_range=FlatDark_KL, dark_range=ObsDark,
#					  sci_range=[SciRanges[0]], std_range=[StdRanges[0]], path=path, base='nov21',
#					  shift=0.0, dtau=-0.0,level1=True,level2=True, SettingsFile='nirspec_21nov2015.ini',
#					  sci_tname='HD88133', std_tname='HR4024', hold_plots=True)
#
