import pynirspec.pynirspec as pn
reload(pn)
#import pynirspec as pn
#reload(pn)

#data_path = '../NIRSPEC_PMC/2017_11_03/' # location of raw data
data_path = '../NIRSPEC_PMC/2017_11_03_fixed/' # location of raw data
ini_path = './ini-files/nirspec_2017nov03_DHTau.ini' # path to initialization file (settings, wranges, and yranges)
output_path = '../pynirspec-out/'

sci_tname = 'DHTau' # name of target
target_type = 'bright'
ut_date = '2017_11_03' # date of observation
base = 'nov03' # base of file name

Flat_K = [(21,140)] # flats
FlatDark_K = [(11,20)] # darks for flats
ObsDark = [(225,228)] # darks for science images
SciRanges = [(193,196)] #  range of science images
#SciRanges = [(193,194),(195,196)] # nod by nod
StdRanges = [(189,192)] # range of standard star

stdname = ['PSO_std','PSO_std','PSO_std','PSO_std','PSO_std','PSO_std']

NRanges = len(SciRanges)
print  NRanges

for i in range(0,NRanges):

	# ## To run each AB pair separately, create a different directory for each
	# sci_tname = 'DHTau-pair' + str(i+1)

	print SciRanges[i]

	Red = pn.Reduction(flat_range=Flat_K, flat_dark_range=FlatDark_K, dark_range=ObsDark,
					   sci_range=[SciRanges[i]], std_range=[StdRanges[0]], path=data_path,
					   output_path = output_path, base=base,shift=0.0, dtau=-0.0,level1=True,level2=False,
					   SettingsFile=ini_path, ut_date = ut_date, sci_tname=sci_tname, target_type = target_type,
					   std_tname=stdname, hold_plots=False)

