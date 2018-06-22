import matplotlib.pylab as plt
import pynirspec.pynirspec as pn
import pyspec as ps
reload(pn)

path = '/home/gablakers/NirspecData/2015_04_06/spec/'
Flat_K = [(248,348)] 
FlatDark_K = [(111,115)]  
ObsDark = [(238,242)] #Darks to not match observations

## HD 88133
#SciRanges = [(181,232)]
#SciRanges = [(181,182)]
SciRanges = [(181, 182), (183, 184), (185, 186), (187, 188), (189, 190), (191, 192), (193, 194),
                      (195, 196), (197, 198), (199, 200), (201, 202), (203,204), (205, 206), (207, 208),
                      (209, 210), (211, 212), (213, 214), (215, 216), (217, 218), (219, 220), (221, 222),
                      (223, 224), (225, 226), (227, 228), (229, 230), (231, 232)]
StdRanges = [(173,180)]


#Red = pn.Reduction(flat_range=Flat_K, flat_dark_range=FlatDark_K, dark_range=ObsDark,
#   	           sci_range=SciRanges, std_range=StdRanges, path=path, base='apr06',
#        	       shift=0.0, dtau=-0.0,level1=True,level2=False, SettingsFile='nirspec_06apr2015.ini',
#            	   sci_tname='HD187123', std_tname='HIP66249', hold_plots=False)


NRanges = len(SciRanges)
for i in range(0,NRanges):
	Red = pn.Reduction(flat_range=Flat_K, flat_dark_range=FlatDark_K, dark_range=ObsDark,
						sci_range=SciRanges[i], std_range=StdRanges, path=path, base='apr06',
						shift=0.0, dtau=-0.0,level1=True,level2=False, SettingsFile='nirspec_06apr2015.ini',
						sci_tname='HD187123', std_tname='HIP66249', hold_plots=False)


	print ''
	print '#########################'
	print '##### Range = ', i+1, ' Completed #####'
	print '#########################'
	print ''
