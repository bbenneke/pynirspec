import pynirspec.pynirspec as pn
reload(pn)
import numpy as np

#data_path = '../NIRSPEC_PMC/2017_11_03/' # location of raw data
data_path = '../NIRSPEC_PMC/2017_11_03_fixed/'
ini_path = './ini-files/nirspec_2017nov03_DHTaub.ini' # path to initialization file (settings, wranges, and yranges)
output_path = '../pynirspec-out/'

sci_tname = 'DHTaub-test' # name of target
target_type = 'faint' ## Faint companions use the Gaussian trace finder to rectify
ut_date = '2017_11_03' # date of observation
base = 'nov03' # base of file name

Flat_K = [(21,140)] # flats
FlatDark_K = [(11,20)] # darks for flats
ObsDark = [(225,228)] # darks for science images
#SciRanges = [(197,198),(199,200),(201,202),(203,204),(205,206),(207,208),(209,210),(211,212)] # Nod by nod
SciRanges = [(197,212)]

StdRanges = [(189,192)] # range of standard star

stdname = ['PSO_std','PSO_std','PSO_std','PSO_std','PSO_std','PSO_std']

NRanges = len(SciRanges)
print  NRanges

for i in range(0,NRanges):
	#print i
	Red = pn.Reduction(flat_range=Flat_K, flat_dark_range=FlatDark_K, dark_range=ObsDark,
					   sci_range=[SciRanges[i]], std_range=[StdRanges[0]], path=data_path, output_path = output_path,
					   base=base,shift=0.0, dtau=-0.0,level1=True,level2=False,
					   SettingsFile=ini_path, ut_date = ut_date, sci_tname=sci_tname, target_type = target_type,
					   std_tname=stdname, hold_plots=False)

## Running 8 sets, where each set omits 1 AB pair
# for i in range(NRanges):
#
# 	if i in [5,6]:
#
# 		print ('set: ' + str(i+1))
# 		print ('-------------------')
# 		sci_range = SciRanges[i]
# 		print ('Using the following '+str(len(sci_range))+' images: ')
# 		print (sci_range)
# 		print ('-------------------')
#
# 		sci_tname_set = sci_tname + '-singleSet' + str(i+1)
#
# 		Red = pn.Reduction(flat_range=Flat_K, flat_dark_range=FlatDark_K, dark_range=ObsDark,
# 							   sci_range=[sci_range], std_range=[StdRanges[0]], path=data_path, output_path = output_path,
# 							   base=base,shift=0.0, dtau=-0.0,level1=True,level2=False,
# 							   SettingsFile=ini_path, ut_date = ut_date, sci_tname=sci_tname_set,
# 							   std_tname=stdname, hold_plots=False, incont_fnum = True)

## Running 8 sets, where each set consists of only 1 AB pair
# for i in range(0, NRanges):
##
# 		print ('set: ' + str(i+1))
# 		print ('-------------------')
# 		sci_range = np.delete(SciRanges, i, axis=0)
# 		sci_range = np.concatenate(sci_range, axis=0)
# 		print ('Using the following '+str(len(sci_range))+' images: ')
# 		print (sci_range)
# 		print ('-------------------')
#
# 		sci_tname_set = sci_tname + '-set' + str(i+1)
#
# 		Red = pn.Reduction(flat_range=Flat_K, flat_dark_range=FlatDark_K, dark_range=ObsDark,
# 							   sci_range=[sci_range], std_range=[StdRanges[0]], path=data_path, output_path = output_path,
# 							   base=base,shift=0.0, dtau=-0.0,level1=True,level2=False,
# 							   SettingsFile=ini_path, ut_date = ut_date, sci_tname=sci_tname_set,
# 							   std_tname=stdname, hold_plots=False, incont_fnum = True)