import matplotlib.pylab as plt
import pynirspec.pynirspec as pn
import pyspec as ps
reload(pn)

path = '/Users/ncrockett/nirspec/data/2013B_dec/Rawdata/'
Flat_coice = [(1,30)]
FlatDark   = [(31,40)]
ObsDark_60x1 = [(51,55)]
ObsDark_10x6 = [(41,45)]

##### SETTING - CO_ICE #####

## LkHa 330, slit = 210 - 30 deg
Red = pn.SAReduction(flat_range=Flat_coice, flat_dark_range=FlatDark, dark_range=ObsDark_60x1,
                     sci_range1=[(85,96)], sci_range2=[(73,84)],std_range=[(97,100)], path=path, base='dec23',
                     shift1=0.0, dtau1=-0.0,shift2=0.0,dtau2=-0.0,level1=True,level2=True, SettingsFile='nirspec_down.ini',
                     sci_tname='LkHa330', std_tname='HR1177', hold_plots=True)
