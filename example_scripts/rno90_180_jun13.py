import matplotlib.pylab as plt
import pynirspec.pynirspec as pn
import pyspec as ps

path = '/astro/pontoppi/DATA/nirspec/2013A_jun/Rawdata/'

Red = pn.SAReduction(flat_range=(21,40), flat_dark_range=(41,45), dark_range=(51,55),
                     sci_range1=(140,147),sci_range2=(148,155),std_range=(112,115), path=path, base='jun24',
                     shift1=0.0, dtau1=-0.0,shift2=0.0,dtau2=-0.0,level1=True,level2=True)

