import matplotlib.pylab as plt
import pynirspec.pynirspec as pn
import pyspec as ps
reload(pn)

path = '/home/gablakers/NirspecData/2015nov21/spec/'
Flat_KL = [(239,366)] 
FlatDark_KL = [(129,133)]  
ObsDark = [(230,232)] #Darks to not match observations

## HD 88133
#SciRanges = [(198,217)]
SciRanges = [(142, 143), (144, 145), (146, 147), (148, 149), (150, 151), (152, 153), (154, 155),
                      (156, 157), (198, 199), (200, 201), (202, 203), (204,205), (206, 207), (208, 209),
                      (210, 211), (212, 213), (214, 215), (216, 217)]
StdRanges = [(218,221)]

NRanges = len(SciRanges)
for i in range(0,NRanges):
	Red = pn.Reduction(flat_range=Flat_KL, flat_dark_range=FlatDark_KL, dark_range=ObsDark,
    	               sci_range=[SciRanges[i]], std_range=StdRanges, path=path, base='nov21',
        	           shift=0.0, dtau=-0.0,level1=True,level2=False, SettingsFile='nirspec_21nov2015.ini',
            	       sci_tname='HD88133', std_tname='HR4024', hold_plots=False)

