import pyfits
import matplotlib.pyplot as plt
import numpy
import sys
from scipy import interpolate 

class EchOrder():
	def __init__(self, FitsFilename):
		self.Mol = {'WLmol': numpy.array([], dtype='float'), 'IDmol':numpy.array([], dtype='str'), 'Tmol':numpy.array([], dtype='str')}
		
	def getFilename(self):
		return self.FitsFilename 

	def getData(self):
		return self.data

	def getMol(self):
		return self.Mol

	def getObsDate(self):
		return self.ObsDate

	def getMinMax(self):
		return (self.WLMin, self.WLMax)

	def readObsDate(self, FitsFilename):
		date_str = FitsFilename.split('_')[-3]
		Date = date_str[4:6] + '/' + date_str[6:8] + '/' + date_str[0:4]		
		return Date

	def readMolecTrans(self, MolFilename):
		MInfo = numpy.loadtxt(MolFilename, dtype='str')
		wl_mt    = numpy.array(MInfo[:,0], dtype='float')
		id_mt    = numpy.array(MInfo[:,1], dtype='str')
		trans_mt = numpy.array(MInfo[:,2], dtype='str')

		sec = numpy.where((wl_mt < self.WLMax) & (wl_mt > self.WLMin)) 
				
		if (len(sec[0]) == 0):
			print 'No Transitions in Order'
		else:
			wl_mt_oi, id_mt_oi, trans_mt_oi = wl_mt[sec], id_mt[sec], trans_mt[sec]
			self.Mol = {'WLmol':wl_mt_oi, 'IDmol':id_mt_oi, 'Tmol':trans_mt_oi}

############################
############################

class SAOrder(EchOrder):
	def __init__(self, FitsFilename):
		self.Mol = {'WLmol': numpy.array([], dtype='float'), 'IDmol':numpy.array([], dtype='str'), 'Tmol':numpy.array([], dtype='str')}
		self.data = self.readFits(FitsFilename)
		self.FitsFilename = FitsFilename
		self.WLMin = min(self.data['WLdata'])
		self.WLMax = max(self.data['WLdata'])
		self.ObsDate = self.readObsDate(FitsFilename)

	def readFits(self, FitsFilename):
		hdulist = pyfits.open(FitsFilename)
		data = hdulist[1].data
		npix = data.shape[0]
		
		wl = numpy.array([])
		flux, uflux = numpy.array([]), numpy.array([])
		sa, usa = numpy.array([]), numpy.array([])
		
		for data_oi in data:
			wl  = numpy.append(wl, data_oi[0])
			flux, uflux = numpy.append(flux, data_oi[1]), numpy.append(uflux, data_oi[2])
			sa, usa     = numpy.append(sa, data_oi[3]), numpy.append(usa, data_oi[4])

		return {'WLdata':wl, 'Fdata':flux, 'UFdata':uflux, 'SAdata':sa, 'USAdata':usa}

############################
############################

class WaveOrder(EchOrder):
	def __init__(self, FitsFilename):
		self.Mol = {'WLmol': numpy.array([], dtype='float'), 'IDmol':numpy.array([], dtype='str'), 'Tmol':numpy.array([], dtype='str')}
		self.data = self.readFits(FitsFilename)
		self.FitsFilename = FitsFilename
		self.WLMin = min(self.data['wl_pos'])
		self.WLMax = max(self.data['wl_pos'])
		self.ObsDate = self.readObsDate(FitsFilename)

	def readFits(self, FitsFilename):
		hdulist = pyfits.open(FitsFilename)
		data = hdulist[1].data
		npix = data.shape[0]
		ra, dec, jd=hdulist[0].header['RA'], hdulist[0].header['DEC'], hdulist[0].header['MJD-OBS']
		air_mass=hdulist[0].header['AIRMASS']
		
		wl_pos, wl_neg = numpy.array([]), numpy.array([])
		flux_pos, uflux_pos, flux_neg, uflux_neg = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
		sky_pos, usky_pos, sky_neg, usky_neg = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
		sa_pos, usa_pos, sa_neg, usa_neg = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
		
		for data_oi in data:
			wl_pos  = numpy.append(wl_pos, data_oi[0])
			flux_pos, uflux_pos = numpy.append(flux_pos, data_oi[1]), numpy.append(uflux_pos, data_oi[2])
			sky_pos, usky_pos = numpy.append(sky_pos, data_oi[3]), numpy.append(usky_pos, data_oi[4])
			sa_pos, usa_pos = numpy.append(sa_pos, data_oi[5]), numpy.append(usa_pos, data_oi[6])

			wl_neg  = numpy.append(wl_neg, data_oi[7])
			flux_neg, uflux_neg = numpy.append(flux_neg, data_oi[8]), numpy.append(uflux_neg, data_oi[9])
			sky_neg, usky_neg = numpy.append(sky_neg, data_oi[10]), numpy.append(usky_neg, data_oi[11])
			sa_neg, usa_neg = numpy.append(sa_neg, data_oi[12]), numpy.append(usa_neg, data_oi[13])

		return {'wl_pos':wl_pos, 'flux_pos':flux_pos, 'uflux_pos':uflux_pos, 'sky_pos':sky_pos, 'usky_pos':usky_pos,
				'sa_pos':sa_pos, 'usa_pos':usa_pos, 'wl_neg':wl_neg, 'flux_neg':flux_neg, 'uflux_neg':uflux_neg,
				'sky_neg':sky_neg, 'usky_neg':usky_neg, 'sa_neg':sa_neg, 'usa_neg':usa_neg, 'ra':ra, 'dec':dec,
				'jd':jd,'airmass':air_mass}

		#return {'WLdata':wl, 'Fdata':flux, 'UFdata':uflux, 'SAdata':sa, 'USAdata':usa}

############################
############################

def PickErrName(yname):
	if (yname == 'sa' or yname == 'sa_norm'):
		output = 'usa'
	if (yname == 'flux' or yname == 'flux_norm'):
		output = 'uflux'
	if (yname == 'Fdata'):
		output = 'UFdata'
	if (yname == 'SAdata'):
		output = 'USAdata'

	return output

############################
############################

def PlotSA(FitsFilenameList=None, MolFilename='mol_trans.dat', Yname='Fdata'):
	nx = 1
	ny = 2
	yrange_fac = 0.3

	ErrYname = PickErrName(Yname)

	## READ IN THE SA FILES ##
	fig, ax = plt.subplots(ny, nx, figsize=(13,8))
	plt.title('test')

	OrderList = []
	i = 0
	for FitsFilename in FitsFilenameList:
		OrderList.append(SAOrder(FitsFilename))
		OrderList[i].readMolecTrans(MolFilename)
		Data = OrderList[i].getData()
		Mol = OrderList[i].getMol()
		WLmol, IDmol, Tmol = Mol['WLmol'], Mol['IDmol'], Mol['Tmol']

		Xvec = Data['WLdata']
		Yvec = Data[Yname]
		ax[i].plot(Xvec, Yvec)
		ax[i].set_ylabel(Yname)

		minY, maxY = min(Yvec[numpy.logical_not(numpy.isnan(Yvec))]), max(Yvec[numpy.logical_not(numpy.isnan(Yvec))])
		rangeY = maxY - minY
		minY_fp, maxY_fp = minY-rangeY*yrange_fac, maxY+rangeY*yrange_fac
		ax[i].set_ylim([minY_fp, maxY_fp])
		if (i == 0):
			ax[i].set_title(FitsFilename)

		nmol = len(Mol['WLmol'])
		for j in range(0,nmol):
			yvec_mol = [numpy.median(Yvec), maxY_fp*0.8]
			xvec_mol = [WLmol[j], WLmol[j]]

			ax[i].plot(xvec_mol, yvec_mol, color='red')
			ax[i].text(WLmol[j], maxY_fp*0.99, IDmol[j]+' '+Tmol[j], ha='center', va='top', size=10, rotation=90)

		i = i+1

	plt.show(block=False)


############################
############################

def PlotWAVE(FitsFilenameList=None, StandFilenameList=None, dtau=0.0, MolFilename='mol_trans.dat', Xname='wl_pos', Yname='flux_pos'):
	nx = 1
	ny = 2
	yrange_fac = 0.3

	#ErrYname = PickErrName(Yname)

	## READ IN THE SA FILES ##
	fig, ax = plt.subplots(ny, nx, figsize=(13,8))

	OrderList = []
	S_OrderList = []
	i = 0
	for FitsFilename in FitsFilenameList:
		OrderList.append(WaveOrder(FitsFilename))
		OrderList[i].readMolecTrans(MolFilename)
		Data = OrderList[i].getData()
		Mol = OrderList[i].getMol()
		WLmol, IDmol, Tmol = Mol['WLmol'], Mol['IDmol'], Mol['Tmol']

		Xvec = Data[Xname]
		Yvec = Data[Yname]
		ax[i].plot(Xvec, Yvec)

		if (StandFilenameList != None):
			DataStand = WaveOrder(StandFilenameList[i]).getData()

			XvecStand = DataStand[Xname]
			YvecStand = DataStand[Yname]
			## Beers Correction
			YvecStand = numpy.exp((1.+dtau)*numpy.log(YvecStand))

			Cont = numpy.median(Yvec)
			ContStand = numpy.median(YvecStand)
			NormFac = Cont / ContStand
			YvecStandNorm = YvecStand*NormFac

			## Plot
			ax[i].plot(XvecStand, YvecStandNorm)

			minY, maxY = min(Yvec[numpy.logical_not(numpy.isnan(Yvec))]), max(Yvec[numpy.logical_not(numpy.isnan(Yvec))])
			rangeY = maxY - minY
			minY_fp, maxY_fp = minY-rangeY*yrange_fac, maxY+rangeY*yrange_fac
			ax[i].set_ylim([minY_fp, maxY_fp])
			ax[i].set_ylabel(Yname)
			if (i == 0):
				ax[i].set_title(FitsFilename)

		nmol = len(Mol['WLmol'])
		for j in range(0,nmol):
			yvec_mol = [numpy.median(Yvec), maxY_fp*0.8]
			xvec_mol = [WLmol[j], WLmol[j]]

			ax[i].plot(xvec_mol, yvec_mol, color='red')
			ax[i].text(WLmol[j], maxY_fp*0.99, IDmol[j]+' '+Tmol[j], ha='center', va='top', size=10, rotation=90)

		i = i+1

	plt.show(block=False)

############################
############################

def PlotCAL(FitsFilenameList=None, MolFilename='mol_trans.dat', Yname='Fdata'):
	nx = 1
	ny = 2
	yrange_fac = 0.3

	ErrYname = PickErrName(Yname)

	## READ IN THE SA FILES ##
	fig, ax = plt.subplots(ny, nx, figsize=(13,8))

	OrderList = []
	i = 0
	for FitsFilename in FitsFilenameList:
		OrderList.append(SAOrder(FitsFilename))
		OrderList[i].readMolecTrans(MolFilename)
		Data = OrderList[i].getData()
		Mol = OrderList[i].getMol()
		WLmol, IDmol, Tmol = Mol['WLmol'], Mol['IDmol'], Mol['Tmol']

		Xvec = Data['WLdata']
		Yvec = Data[Yname]
		ax[i].plot(Xvec, Yvec)

		minY, maxY = min(Yvec[numpy.logical_not(numpy.isnan(Yvec))]), max(Yvec[numpy.logical_not(numpy.isnan(Yvec))])
		rangeY = maxY - minY
		minY_fp, maxY_fp = minY-rangeY*yrange_fac, maxY+rangeY*yrange_fac
		ax[i].set_ylim([minY_fp, maxY_fp])

		nmol = len(Mol['WLmol'])
		for j in range(0,nmol):
			yvec_mol = [numpy.median(Yvec), maxY_fp*0.8]
			xvec_mol = [WLmol[j], WLmol[j]]

			ax[i].plot(xvec_mol, yvec_mol, color='red')
			ax[i].text(WLmol[j], maxY_fp*0.99, IDmol[j]+' '+Tmol[j], ha='center', va='top', size=10, rotation=90)

		i = i+1

	plt.show(block=False)




