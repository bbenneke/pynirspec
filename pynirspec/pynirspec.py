import warnings
import json
import os
import ConfigParser as cp
import time as t

import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
import scipy.fftpack as fp
from scipy.stats import tmean, tvar
from scipy.ndimage.filters import median_filter
from scipy import constants
import matplotlib.pylab as plt
from collections import Counter
import sys
import inpaint as inpaint
import matplotlib.cm as cm
sys.path.insert(0, '../atmopt')
import rfm_tools as rfm
#test
class Environment():
    '''
    Class to encapsulate global environment parameters.

    Parameters
    ----------
    config_file: string
        ConfigParser compliant file containing global parameters.

    Methods
    -------
    
    Attributes
    ----------
    pars: SafeConfigParser object
        Contains all global parameters

    '''
    def __init__(self,settings_file=None,detpars_file=None):

        sys_dir = self._getSysPath()
        #print ("sys_dir: " + sys_dir)

        self.settings = cp.SafeConfigParser()
        if settings_file is None:
            #use the system settings
            self.settings.read(sys_dir+'/'+'nirspec.ini')
        else:
            self.settings.read(settings_file)
            
        self.detpars = cp.SafeConfigParser()
        if detpars_file is None:
            self.detpars.read(sys_dir+'/'+'detector.ini')
        else:
            self.detpars.read(detpars_file)

    def _getSysPath(self):
        sys_dir, this_filename = os.path.split(__file__)
        return sys_dir

    def getItems(self,option):
        return [self.settings.get(section,option) for section in self.settings.sections()]

    def getSections(self):
        return self.settings.sections()

    def getWaveRange(self,setting,onum):
        range_str = self.settings.get(setting,'wrange'+str(int(onum)))
        range = json.loads(range_str)
        return range
        
    def getYRange(self,setting,onum):
        range_str = self.settings.get(setting,'yrange'+str(int(onum)))
        range = json.loads(range_str)
        return range

    def getDispersion(self,setting,onum):
        A_str = self.settings.get(setting,'A'+str(int(onum)))
        B_str = self.settings.get(setting,'B'+str(int(onum)))
        C_str = self.settings.get(setting,'C'+str(int(onum)))
        R_str = self.settings.get(setting,'R'+str(int(onum)))
        As = json.loads(A_str)
        Bs = json.loads(B_str)
        Cs = json.loads(C_str)
        Rs = json.loads(R_str)
        return {'A':As,'B':Bs,'C':Cs,'R':Rs}

    # Returns a dict containing detector gain, read noise, dark current
    def getDetPars(self):
        gain = self.detpars.getfloat('Detector','gain')
        rn   = self.detpars.getfloat('Detector','rn')
        dc   = self.detpars.getfloat('Detector','dc')
        return {'gain':gain,'rn':rn,'dc':dc}

    # How many orders of spectra
    def getNOrders(self,setting):
        return self.settings.getint(setting,'norders')

class Observation():
    '''
    Private object containing a NIRSPEC observation - that is, all exposures related to a single type of activity.

    Any specific activity (Darks, Flats, Science, etc.) are modeled as classes derived off the Observation class.

    Parameters
    ----------
    filelist: List of strings
        List of data (.fits) files associated with the observation.
    type: string
        type of observation (e.g., Dark). 

    Attributes
    ----------
    type    # darks/flats/sci etc.
    Envi    # instrument settings and observation details (as contained in the .ini files)
    flist   # file list
    planes  # list of opened fits files: e.g. planes[0] is the first file, and planes[0][0] is the PrimaryHDU of the fits file
    header  # header of a specific fits file

    Methods
    -------
    getSetting
    getNOrders
    getTargetName
    subtractFromStack
    divideInStack
    writeImage
   
    '''
    def __init__(self,filelist,type='image', SettingsFile=None,tname=None,ut_date=None):
        self.type = type
        self.Envi = Environment(settings_file=SettingsFile)
        self.flist = filelist
        self._openList(tname)
        self._makeHeader(tname)
        self.ut_date = ut_date
        self.sci_tname = tname

    # Open the files and get exp and det parameters from header and .ini file respectively
    def _openList(self, tname):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)

        # Construct a list of opened fits files (HDU lists)
        self.planes = []
        for file in self.flist:
            plane = pf.open(file,ignore_missing_end=True)
            self.planes.append(plane)
        self._makeHeader(tname)

        # Get exposure parameters from header
        self.exp_pars = self.getExpPars()
        # Get detector parameters from .ini files
        self.det_pars = self.Envi.getDetPars()

    # Extract header info from the first file
    def _makeHeader(self, tname=None):
        self.header = self.planes[0][0].header 
        
        #The NIRSPEC headers have a couple of illegally formatted keywords
        try:
            del self.header['GAIN.SPE']
        except:
            print ('GAIN.SPE already deleted from hdr.')
        try:
            del self.header['FREQ.SPE']
        except:
            print ('FREQ.SPE already deleted from hdr.')

        # Replace object name in header with custom supplied name
        if (tname is not None):
            self.header['OBJECT'] = tname.replace(' ','')

    # Returns a dict of exposure info
    def getExpPars(self):
        coadds   = self.getKeyword('COADDS')[0]
        sampmode = self.getKeyword('SAMPMODE')[0]
        nreads   = self.getKeyword('MULTISPE')[0]
        itime    = self.getKeyword('ITIME')[0]
        # Number of images
        nexp     = len(self.planes)
        return {'coadds':coadds,'sampmode':sampmode,'nreads':nreads,'itime':itime,'nexp':nexp}

    # Return correct setting name, echelle pos, and crossdisp pos
    def getSetting(self):
        echelle   = self.getKeyword('ECHLPOS')
        crossdisp = self.getKeyword('DISPPOS')

        # Check that all exposures are taken with the same echelle position and cross disperser position
        assert len([e for e in echelle if e==echelle[0]])==len(echelle), \
            'All exposures must be taken with the same setting!'
        assert len([c for c in crossdisp if c==crossdisp[0]])==len(crossdisp), \
            'All exposures must be taken with the same setting!'

        echelle   = echelle[0]
        crossdisp = crossdisp[0]

        #print ('echelle: ' + str(echelle))
        #print ('crossdisp: ' + str(crossdisp))

        echelles   = self.Envi.getItems('echelle')
        crossdisps = self.Envi.getItems('crossdisp')

        # print (echelles)
        # print (crossdisps)

        # Note: the following steps are unnecessary when user supplies a custom .ini file
        # Get indices of echelle position in .ini file that match that in the header
        setsub1 = [i for i,v in enumerate(echelles) if float(v)==echelle]
        # print (setsub1)
        # Get indices of crossdisp position in .ini file that match that in the header
        setsub2 = [i for i,v in enumerate(crossdisps) if float(v)==crossdisp]
        # print (setsub2)
        # Determine the correct section (configuration) as
        # the one containing both the correct echelle pos and crossdisp pos
        sub = [i for i in setsub1 if i in setsub2]
        # print (sub)
        #print (self.Envi.getSections()[sub[0]])
        return self.Envi.getSections()[sub[0]],echelle,crossdisp

    # Return a list of airmasses, converting faulty values to the mean
    def getAirmass(self):
        airmasses = self.getKeyword('AIRMASS') 
        airmasses = np.array([airmass if type(airmass) is not str else -1 for airmass in airmasses])
        mean = np.mean(airmasses[np.where(airmasses!=-1)])
        airmasses[np.where(airmasses==-1)] = mean
        return airmasses

    # Get number of orders
    def getNOrders(self):
        setting,echelle,crossdisp = self.getSetting()
        return self.Envi.getNOrders(setting)

    def getTargetName(self):
        target_name = self.header['OBJECT']
        return target_name

    # Prepare data stack in units of e- counts, and an error stack
    def _getStack(self):
        
        nexp = self.exp_pars['nexp']
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']
        
        stack = np.zeros((nx,ny,nexp))
        ustack = np.zeros((nx,ny,nexp))
        
        for i,plane in enumerate(self.planes):
            # data stack (counts*gain = e- count)
            stack[:,:,i]  = plane[0].data*self.det_pars['gain'] #convert everything to e-
            # error stack
            ustack[:,:,i] = self._error(plane[0].data)
        return stack,ustack

    # Compute uncertainty as sqrt of variance
    def _error(self,data):
        var_data = np.abs(data+self.exp_pars['itime']*self.det_pars['dc']+
                          self.det_pars['rn']**2/self.exp_pars['nreads'])
        return np.sqrt(var_data)

    # Method to subtract darks from flats
    def subtractFromStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
        try:
            self.uimage = np.sqrt(self.uimage**2+Obs.uimage**2)
            self.image -= Obs.image
        except:
            print 'Subtraction failed - no image calculated'

    # Flat-field correction
    def divideInStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]
            uplane = self.ustack[:,:,i]
            uplane = np.sqrt((uplane/plane)**2+(Obs.uimage/Obs.image)**2)
            # Divide data by normalized flats
            plane /= Obs.image
            uplane *= np.abs(plane)

            self.stack[:,:,i]  = plane
            self.ustack[:,:,i] = uplane

    # Collapse darks and flats sequences into single images, with a weighted average (assuming Gaussian error)
    def _collapseStack(self,stack=None,ustack=None,method='SigClip',sig=50.):
        '''
        If called without the stack keyword set, this will collapse the entire stack.
        However, the internal stack is overridden if a different stack is passed.
        For instance, this could be a stack of nod pairs.
        '''
        if stack is None:
            stack,ustack = self.stack,self.ustack

        #stack_median = np.median(stack,2)
        #stack_stddev = np.std(stack,2)
        #shape = stack.shape
        #masked_stack = ma.zeros(shape)

        # Mask invalid points so that errors do not occur in math operations
        masked_stack = ma.masked_invalid(stack)
        masked_ustack = ma.masked_invalid(ustack)

        # print (masked_stack.shape)
        # print (masked_ustack.shape)

        # Take weighted average of a sequence of images
        image = ma.average(masked_stack,2,weights=1./masked_ustack**2)
        uimage = np.sqrt(ma.mean(masked_ustack**2,2)/ma.count(masked_ustack,2))

        # print (image.shape)
        # print (uimage.shape)

        return image, uimage

    # Save image (data + uncertainty)
    def writeImage(self,filename=None):
        if filename is None:
            filename = self.type+'.fits'
            
        hdu = pf.PrimaryHDU(self.image.data)
        uhdu = pf.ImageHDU(self.uimage.data)
        hdulist = pf.HDUList([hdu,uhdu])
        hdulist.writeto(filename,clobber=True)

    # Returns list of values for a specific header key
    def getKeyword(self,keyword):
        try:
            klist = [plane[0].header[keyword] for plane in self.planes]
            return klist
        except ValueError:
            print "Invalid header keyword"

## Flat class, extends the class Observation
class Flat(Observation):
    def __init__(self,filelist,dark=None,norm_thres=5000.,save=False,ut_date=None,sci_tname=None,**kwargs):

        # Inherits __init__ function from Observation class
        Observation.__init__(self,filelist,ut_date=ut_date,tname=sci_tname,**kwargs)
        self.type = 'flat'
        # Convert images to units of e- count
        self.stack,self.ustack = self._getStack()
        # Number of flats
        self.nplanes = self.stack.shape[2]
        # Frame size of flats
        self.height = self.stack[:,:,0].shape[0]
        # Get setting parameters
        self.setting,self.echelle,self.crossdisp = self.getSetting()
        # Combine flats through a weighted average
        self.image,self.uimage = self._collapseStack()

        # Whether to subtract flat darks from flats
        if dark:
            self.subtractFromStack(dark)
        # Normalize flats by median
        self._normalize(norm_thres)

        # Where the flat field is faulty, it's set to 1 to avoid divide by zeros.
        self.image[np.where(self.image<0.1)] = 1
        if save:
            self.writeImage(filename='../pynirspec-out/' + self.ut_date + '/' + self.sci_tname + '/' + self.type + '.fits')

    # Normalize flats with median value (larger than a given threshold)
    def _normalize(self,norm_thres):
        flux = np.median(self.image[np.where(self.image>norm_thres)])
        self.image = self.image/flux
        self.uimage = self.uimage/flux
    # Not used
    def makeMask(self):
        return np.where(self.image<0.1,0,1)

## Dark class, extends the class Observation
class Dark(Observation):
    def __init__(self,filelist,save=False,ut_date=None,sci_tname=None,**kwargs):
        # Inherits __init__ function from Observation class
        Observation.__init__(self,filelist,ut_date=ut_date,tname=sci_tname,**kwargs)
        self.type = 'dark'
        # Convert images to units of e- count
        self.stack,self.ustack = self._getStack()
        # Number of darks
        self.nplanes = self.stack.shape[2]
        # Frame size of darks
        self.height = self.stack[:,:,0].shape[0]
        # Combine darks using a weighted average
        self.image,self.uimage = self._collapseStack()
        # Make bad pixel map (see function below)
        self._badPixMap(filename='../pynirspec-out/' + self.ut_date + '/' + self.sci_tname + '/badpix.dmp')
        # Whether to save combined dark frame
        if save:
            self.writeImage(filename='../pynirspec-out/' + self.ut_date + '/' + self.sci_tname + '/' + self.type + '.fits')

    # Create bad pixel map
    def _badPixMap(self,clip=30,filename='badpix.dmp'):
        median = np.median(self.image)
        # Variance is computed for values between -100 and 100 - why a range???
        var  = tvar(self.image,(-100,100))
        # Bad pixels are defined as those where the count-median is greater than 30*Sqrt(variance)
        self.badpix = ma.masked_greater(self.image-median,clip*np.sqrt(var))

        # Save the bad pixel map as a numpy masked array
        if filename is not None:
            self.badpix.dump(filename)

## Class for science images (both A and B nods). Prepares A-B images, as well as a sky image
class Nod(Observation):
    def __init__(self,filelist,dark=None,flat=None,badpix='badpix.dmp',**kwargs):

        # Inherits Observation
        Observation.__init__(self,filelist,**kwargs)
        self.type = 'nod'                
        self.setting,self.echelle,self.crossdisp = self.getSetting()
        # Get list of airmasses through the science sequence
        self.airmasses = self.getAirmass()
        # Mean airmass
        self.airmass = np.mean(self.airmasses)

        # List of RA, DEC, and file numbers for science images
        RAs  = self.getKeyword('RA')
        DECs = self.getKeyword('DEC')
        FileNums = self.getKeyword('FILENUM')        

        # Get indices of each AB nod pair, e.g. [(0, 1), (3, 2), (4, 5), (7, 6), (8, 9), (11, 10)]
        pairs = self._getPairs(RAs,DECs,FileNums)
        # Get sequence of A-B images from nod pairs
        self.stack,self.ustack = self._makePairStack(pairs)
        
        # self.stack,self.ustack = self._getStack()  #MLB ADD - these are the A, B images (not A-B)
        # Frame size
        self.height = self.stack[:,:,0].shape[0]

        plt.imshow(self.stack[:, :, 0], vmin=-200, vmax=200, cmap='gray')
        plt.colorbar()
        plt.text(100, 100, '1 A-B BEFORE dividing by flat & bp correction')
        plt.savefig('../pynirspec-out/2017_11_03/sample-A_B-before-flattening.png')

        # Divide by normalized flats
        if flat:
            self.divideInStack(flat)
        # Correct bad pixels through a weighted local mean
        if badpix:
            badmask = np.load(badpix)
            self._correctBadPix(badmask)

        plt.imshow(self.stack[:, :, 0], vmin=-200, vmax=200, cmap='gray')
        plt.colorbar()
        plt.text(100, 100, '1 A-B AFTER dividing by flat & bp correction')
        plt.savefig('../pynirspec-out/2017_11_03/sample-A_B-after-flattening.png')
        # t.sleep(5)
        #plt.close()

       # Reassign variable names for A-B images
        self.TargetStack, self.UTargetStack = self.stack, self.ustack

        # stack_t = self.TargetStack
        # print (type(stack_t))
        # print (stack_t.shape)
        
        # An averaged sky frame is constructed using the off-beam pixels
        # Separated stacks for A images and B images
        stackA,ustackA,stackB,ustackB = self._makeSingleBeamStacks(pairs)
        beamStacks = [(stackA,ustackA),(stackB,ustackB)] 

        # Creates preprocessed (not dark-subtracted???) images for A image and B image separately)
        beam_sky_stacks, beam_usky_stacks = [], []
        test_count = 0
        for beamStack in beamStacks:
            self.stack = beamStack[0]
            self.ustack = beamStack[1]

            if test_count == 0:
                #Plot image before flat correction
                plt.imshow(self.stack[:,:,0], vmin=-200, vmax=1500, cmap='gray')
                plt.colorbar()
                plt.text(100, 100, '1 nod BEFORE dividing by flat & bp correction')
                plt.savefig('../pynirspec-out/2017_11_03/sample-Anod-before-flattening.png')

            if flat:
                self.divideInStack(flat)

            if badpix:
                badmask = np.load(badpix)
                self._correctBadPix(badmask)

            if test_count == 0:
                #Plot flat-corrected image to compare
                plt.imshow(self.stack[:,:,0], vmin=-200, vmax=1500, cmap='gray')
                plt.colorbar()
                plt.text(100, 100, '1 nod AFTER dividing by flat & bp correction')
                plt.savefig('../pynirspec-out/2017_11_03/sample-Anod-after-flattening.png')


            beam_sky_stacks.append(self.stack)
            beam_usky_stacks.append(self.ustack)
            test_count = test_count + 1

        self.beamSkyStacks  = beam_sky_stacks
        self.beamUSkyStacks = beam_usky_stacks

    # Correct bad pixels using an iterative local mean method from inpaint.py
    def _correctBadPix(self,badmask):
        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]            
            maskedImage = np.ma.array(plane, mask=badmask.mask)
            NANMask = maskedImage.filled(np.NaN)
            self.stack[:,:,i] = inpaint.replace_nans(NANMask, method='localmean')          
            
            plane = self.ustack[:,:,i]            
            maskedImage = np.ma.array(plane, mask=badmask.mask)
            NANMask = maskedImage.filled(np.NaN)            
            self.ustack[:,:,i] = inpaint.replace_nans(NANMask,method='localmean')          
            
    # Compute A-B image for each AB nod pair
    def _makePairStack(self,pairs):
        # Number of AB pairs
        npairs = len(pairs)
        # x and y frame sizes
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']

        # Convert sci images to units of e- counts
        stack,ustack = self._getStack()
        pair_stack  = np.zeros((nx,ny,npairs))
        pair_ustack = np.zeros((nx,ny,npairs))

        # Iterate over all pairs to compute A-B for each pair
        for i,pair in enumerate(pairs):
            pair_stack[:,:,i] = stack[:,:,pair[0]] - stack[:,:,pair[1]]
            pair_ustack[:,:,i] = np.sqrt(ustack[:,:,pair[0]]**2 + ustack[:,:,pair[1]]**2)
            
        return pair_stack,pair_ustack

    # Returns stacks for A nods and B nods
    def _makeSingleBeamStacks(self,pairs):
        npairs = len(pairs)
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']

        stack,ustack = self._getStack()
        stackA  = np.zeros((nx,ny,npairs))
        ustackA = np.zeros((nx,ny,npairs))
        stackB  = np.zeros((nx,ny,npairs))
        ustackB = np.zeros((nx,ny,npairs))
        
        for i,pair in enumerate(pairs):
            stackA[:,:,i] = stack[:,:,pair[0]]
            ustackA[:,:,i] = ustack[:,:,pair[0]]
            stackB[:,:,i] = stack[:,:,pair[1]]
            ustackB[:,:,i] = ustack[:,:,pair[1]]
        return stackA,ustackA,stackB,ustackB

    # Determine which images are A nods, and which are B nods
    # Returns a list of arrays, where each array is a AB pair
    # e.g. [(0, 1), (3, 2), (4, 5), (7, 6), (8, 9), (11, 10)] for 12 images in a standard ABBA pattern
    def _getPairs(self,RAs,DECs,FileNums):
        nexp = len(RAs)                
        assert nexp % 2 ==0, "There must be an even number of exposures"

        AorB = []
        RA_A  = RAs[0]
        DEC_A = DECs[0]

        RA_B  = RAs[1]
        DEC_B = DECs[1]

        dist_a = np.sqrt(((np.array(RAs)-RA_A)*3600)**2+((np.array(DECs)-DEC_A)*3600)**2)
        dist_b = np.sqrt(((np.array(RAs)-RA_B)*3600)**2+((np.array(DECs)-DEC_B)*3600)**2)
        
        for i in range(0,nexp):
            if (dist_a[i] < dist_b[i]):
                AorB.append('A')
            else:
                AorB.append('B')

        print ''
        print '%7s %12s %12s %5s' % ('Exp.', 'RA', 'DEC', 'AorB')
        print '%7s %12s %12s %5s' % (str('-'*7), str('-'*12), str('-'*12), str('-'*5))    
        for i in range(0,len(RAs)):
            print  '%7i %12f %12f %5s' % (FileNums[i], RAs[i], DECs[i], AorB[i])
        print '\n'

        #LOOK FOR PROBLEMS IN RA AND DEC FROM HEADERS
        problem = False
        if (0.0 in RAs or 0.0 in DECs):
            problem = True

        TotAorB = Counter((np.array(AorB))) 
        TotA, TotB = TotAorB['A'], TotAorB['B']
        if (TotA != TotB):
            problem = True

        if (problem):
            print 'Warning!! - There may be a problem with the pointing in these exposures.'

            while (True):
                yn_correct = raw_input('Is the AB pattern shown above correct? [yes/no]: ').lower()
                if (yn_correct == 'no'):
                    do_fix = True
                    break
                elif (yn_correct == 'yes'):
                    do_fix = False
                    break
                else:
                    print "Please type 'yes' or 'no'."

            while (do_fix):
                yn_abba = raw_input('Assume an ABBA pattern? [yes/no]: ').lower()
                if (yn_abba == 'yes'):
                    if (nexp % 4 == 0):
                        AorB = 'ABBA'*(nexp/4)
                        do_fix = False
                    else:
                        print 'The number of exposures must be a multiple of 4 to assume an ABBA pattern.'
                elif (yn_abba == 'no'):
                    do_setnod = True
                    while (do_setnod):
                        yn_pattern = raw_input('Set nod pattern? [yes/no]: ').lower()
                        if (yn_pattern == 'yes'):
                            nod_pattern = raw_input('Import pattern, e.g. ABBA BA ABBA \n (case or spaces do not matter): ').upper().replace(" ", "")
                            AorB = list(nod_pattern)
                            do_setnod, do_fix = False, False
                        elif (yn_pattern == 'no'):
                            print 'Please fix image headers and run the pipeline again.'
                            sys.exit()
                        else:
                            print "Please type 'yes' or 'no'."
                else:
                    print "Please type 'yes' or 'no'."

            print''
            print '## Revised AB Pairs ##'
            print '======================'
            print '%7s %12s %12s %5s' % ('Exp.', 'RA', 'DEC', 'AorB')
            print '%7s %12s %12s %5s' % (str('-'*7), str('-'*12), str('-'*12), str('-'*5))    
            for i in range(0,len(RAs)):
                print  '%7i %12f %12f %5s' % (FileNums[i], RAs[i], DECs[i], AorB[i])
            print '\n'
        
        ii = 0
        pairs = []
        while ii<nexp:
            if AorB[ii]!=AorB[ii+1]:
                if AorB[ii]=='A':
                    pair = (ii,ii+1)
                else:
                    pair = (ii+1,ii)
                pairs.append(pair)
                ii+=2
            else:
                ii+=1
        #print (pairs)
        return pairs

## Class to prepare a single order, resulting in a 2D spectral order, saved in SPEC2D/
## 1) create A-B images for a specific order, 2) shift images to match pos of the first image,
## 3) combine images, and 4) rectify the combined image by fitting a polynomial
class Order():
    def __init__(self,Nod,onum=1,trace=None,write_path=None):
        self.type = 'order'
        self.header  = Nod.header
        self.setting = Nod.setting
        self.echelle = Nod.echelle
        self.crossdisp = Nod.crossdisp
        self.airmass = Nod.airmass
        self.Envi    = Nod.Envi
        # Order being processed
        self.onum    = onum
        # Spatial range of order in units of pixels
        self.yrange = self.Envi.getYRange(self.setting,onum)
        
        # Crop A-B images in the spatial direction to extract a single order
        self.stack  = Nod.TargetStack[self.yrange[0]:self.yrange[1],:,:]
        #print (self.stack.shape)
        self.ustack = Nod.UTargetStack[self.yrange[0]:self.yrange[1],:,:]
        nexp = len(self.stack[0,0,:])

        img_stack = self.stack

        # Get y offsets between the same order from each image
        offsets1, offsets2 = self._findYOffsets()
        # Shift based on offsets, so that when combining the images, the order from different images fit each other
        self.stack1  = self._yShift(offsets1,self.stack)
        self.ustack1 = self._yShift(offsets1,self.ustack)
        self.stack2  = self._yShift(offsets2,self.stack)
        self.ustack2 = self._yShift(offsets2,self.ustack)

        # Combine A-B images with weighted average
        self.image1, self.uimage1 = self._collapseOrder(stack=self.stack1, ustack=self.ustack1)
        self.image2, self.uimage2 = self._collapseOrder(stack=self.stack2, ustack=self.ustack2)


        # Fit polynomial to rectify combined A-B image
        if trace is None:
            yr1, trace1 = self.fitTrace(self.image1, 1)
            yr2, trace2 = self.fitTrace(self.image2, 2)
            yrs, traces  = [yr1, yr2], [trace1, trace2]
  
        images, uimages = [self.image1, self.image2], [self.uimage1, self.uimage2]

        # Use trace to correct for non-linearity in combined order, i.e. rectifying
        self.image_rect, self.uimage_rect = self.yRectify(images, uimages, yrs, traces)
        self.sh = self.image_rect.shape

        # Subtract median of each column from the same column - SKY SUBTRACTION METHOD
        self._subMedian()

        # Repeat above for individual A and B images
        self.beamSkyStacks  = Nod.beamSkyStacks  
        self.beamUSkyStacks = Nod.beamUSkyStacks

        beamSkyRect, beamUSkyRect = [], []
        for beamSkyStack, beamUSkyStack in zip(self.beamSkyStacks, self.beamUSkyStacks):
            # Crop sky images in spatial direction to extract a single order
            beamSkyStackO  = beamSkyStack[self.yrange[0]:self.yrange[1],:,:]
            beamUSkyStackO = beamUSkyStack[self.yrange[0]:self.yrange[1],:,:]
            
            beamSkyStackO1  = self._yShift(offsets1, beamSkyStackO)
            beamUSkyStackO1 = self._yShift(offsets1, beamUSkyStackO)
            beamSkyStackO2  = self._yShift(offsets2, beamSkyStackO)
            beamUSkyStackO2 = self._yShift(offsets2, beamUSkyStackO)

            beamSky1, beamUSky1 = self._collapseOrder(stack=beamSkyStackO1, ustack=beamUSkyStackO1)
            beamSky2, beamUSky2 = self._collapseOrder(stack=beamSkyStackO2, ustack=beamUSkyStackO2)

            beamSkies, beamUSkies = [beamSky1, beamSky2], [beamUSky1, beamUSky2]
            beam_sky_rect, beam_usky_rect = self.yRectify(beamSkies, beamUSkies, yrs, traces)
            beamSkyRect.append(beam_sky_rect), beamUSkyRect.append(beam_usky_rect)

        self.sky_rect  = beamSkyRect[0]
        self.usky_rect = beamUSkyRect[0]

        # print ('sky_rect shape')
        # print (self.sky_rect.shape)

        # plt.imshow(self.sky_rect, cmap='gray')
        # plt.colorbar()
        # plt.text(30, 30, 'sky rect initial')
        # plt.show()
        # #t.sleep(5)
        # plt.close()

        ## These lines create the sky! This is done by using values from both the A and B image
        ## Specifically, (B-A) image is made, and where the counts are < 2*error,
        ## the counts from A are replace with the counts from B. This ensures the spectra in A is replaced by sky counts from B
        ssubs = np.where(beamSkyRect[1]-beamSkyRect[0]<2.*beamUSkyRect[0])
        self.sky_rect[ssubs]  = beamSkyRect[1][ssubs]
        self.usky_rect[ssubs] = beamUSkyRect[1][ssubs]

        # plt.imshow(self.sky_rect, cmap='gray')
        # plt.colorbar()
        # plt.text(30, 30, 'sky rect final')
        # plt.show()
        # #t.sleep(5)
        # plt.close()

        # plt.imshow(self.image_rect, cmap='gray')
        # plt.colorbar()
        # plt.text(30, 30, 'SPEC 2D')
        # plt.show()
        # #t.sleep(5)
        # plt.close()

        # plt.imshow(self.uimage_rect, cmap='gray')
        # plt.colorbar()
        # plt.text(30, 30, 'SPEC 2D errors')
        # plt.show()
        # #t.sleep(5)
        # plt.close()

        if write_path:
            self.file = self.writeImage(path=write_path)


    def _collapseOrder(self, stack=None, ustack=None):

        sh = stack.shape
        int_stack = np.zeros(sh)
        int_ustack = np.zeros(sh)
        
        index = np.arange(sh[0])
   
        masked_stack = ma.masked_invalid(stack)
        masked_ustack = ma.masked_invalid(ustack)

        image = ma.average(masked_stack,2,weights=1./masked_ustack**2)
        uimage = np.sqrt(ma.mean(masked_ustack**2,2)/ma.count(masked_ustack,2))
        
        return image, uimage

    # Shifts images in y (spatial direction) so that they are aligned for combining
    def _yShift(self,offsets,stack):

        sh = stack.shape
        internal_stack = np.zeros(sh)

        index = np.arange(sh[0])

        for plane in np.arange(sh[2]):
            for i in np.arange(sh[1]):
                col = np.interp(index-offsets[plane],index,stack[:,i,plane])
                internal_stack[:,i,plane] = col

        return internal_stack

    # Compute offsets in y (spatial direction) between different images
    def _findYOffsets(self, kwidth=100):
        sh = self.stack.shape
        yr1 = (0, sh[0]/2-1)     #Bottom half of order (A or B pos)
        yr2 = (sh[0]/2,sh[0]-1)  #Top half of order (other A or B pos)
        yrs = [yr1,yr2]

        offsets1 = np.empty(0)
        offsets2 = np.empty(0)
        # Iterate twice, for bottom half and then top half of order
        for i in range(0, len(yrs)):
            yr = yrs[i]
            yindex = np.arange(yr[0],yr[1]+1) #Top or Bottom of Order
            # Use the first A-B image as the kernel
            kernel_image = self.stack[:,:,0]
            #print (yindex)

            # Median of values along the wavelength direction, over the central 200 pixels
            kernel_o = np.median(kernel_image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth],1)
            #print ((kernel_image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth]).shape)
            #print (kernel_o)
            # Median of the top or bottom half of the order
            kernel_med = np.median(kernel_o)
            # why subtract, what is the kernel???
            kernel = np.subtract(kernel_o, kernel_med)

            # Iterate over A-B images
            for j in range(0,sh[2]):
                image = self.stack[:,:,j]                
                profile_o = np.median(image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth],1)
                profile_med = np.median(profile_o)
                profile = np.subtract(profile_o, profile_med)
                # Cross correlation between kernel and image - ???
                cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(profile)))
                cc_sh = fp.fftshift(cc)
                cen = calc_centroid(cc_sh).real - yindex.shape[0]/2.

                if (i == 0):
                    offsets1 = np.append(offsets1,cen)
                if (i == 1):
                    offsets2 = np.append(offsets2,cen)

                
        print ''
        print 'Img  A      B'
        print '------'
        for k in range(0,sh[2]):
            print ('%i  %5.2f %5.2f') % (k+1, offsets1[k], offsets2[k])
        print ''

        return offsets1, offsets2

    def fitTrace(self, image, OneOrTwo, kwidth=40):
        sh = image.shape

        if (OneOrTwo == 1):
            yr = (0,sh[0]/2-1)
        if (OneOrTwo == 2):
            yr = (sh[0]/2,sh[0]-1)
        
        yindex = np.arange(yr[0],yr[1]+1) #Top or Bottom of Order
        kernel = np.median(image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth],1)
                    
        centroids = []
        totals = []
        # Iterate over each col (wavelength direction)
        for i in np.arange(sh[1]):
            col_med = np.median(image[yindex,i])
            total = np.abs((image[yindex,i]-col_med).sum())
            cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(image[yindex,i]-col_med)))
            cc_sh = fp.fftshift(cc)
            centroid = calc_centroid(cc_sh).real - yindex.shape[0]/2.
            centroids.append(centroid)
            totals.append(total)

        centroids = np.array(centroids)
        totals = np.array(totals)
        median_totals = np.median(totals)            
        
        xindex = np.arange(sh[1])
        gsubs = np.where((np.isnan(centroids)==False) & (totals>median_totals*0.25) & (totals<median_totals*1.75))

        centroids[gsubs] = median_filter(centroids[gsubs],size=50)
        # Fit 3-order polynomial to order
        coeffs = np.polyfit(xindex[gsubs],centroids[gsubs],3)

        poly = np.poly1d(coeffs)
            
        return yr,poly

    def yRectify(self,images,uimages,yrs,traces): 

        sh = images[0].shape
        image_rect = np.zeros(sh)
        uimage_rect = np.zeros(sh)
        
        for yr,trace,image,uimage in zip(yrs,traces,images,uimages):
            index = np.arange(yr[0],yr[1]+1) 

            for i in np.arange(sh[1]):
                col = np.interp(index-trace(i),index,image[index,i])
                image_rect[index,i] = col
                col = np.interp(index-trace(i),index,uimage[index,i])
                uimage_rect[index,i] = col

        return image_rect,uimage_rect
        
    # Subtract median of each column from that column
    def _subMedian(self):
        #print (self.image_rect[:,500])
        #print (np.median(self.image_rect, axis=0)[500])
        self.image_rect = self.image_rect-np.median(self.image_rect,axis=0)
        #print ((self.image_rect).shape)
        #print (self.image_rect[:,500])

    def writeImage(self,filename=None,path='.'):

        if filename is None:
            time   = self.header['UTC']
            time   = time.replace(':','')
            time   = time[0:4]
            date   = self.header['DATE-OBS']
            date   = date.replace('-','')
            object = self.header['OBJECT']
            object = object.replace(' ','')
            filename = path+'/'+object+'_'+date+'_'+time+'_order'+str(self.onum)+'.fits'

            
        hdu  = pf.PrimaryHDU(self.image_rect)
        uhdu = pf.ImageHDU(self.uimage_rect)
        sky_hdu = pf.ImageHDU(self.sky_rect)
        usky_hdu = pf.ImageHDU(self.usky_rect)

        #print (type(hdu.data))

        hdu.header['SETNAME'] = (self.setting, 'Setting name')
        hdu.header['ECHLPOS'] = (self.echelle, 'Echelle position')
        hdu.header['DISPPOS'] = (self.crossdisp, 'Cross disperser position')
        hdu.header['ORDER'] = (str(self.onum),'Order number')

        hdulist = pf.HDUList([hdu,uhdu,sky_hdu,usky_hdu])

        hdulist.writeto(filename,clobber=True)

        return filename

## Class to extract 1D spectra for positive and negative parts of an order respectively
## Returns spectra from positive and negative part, the associated errors, and the spectra of the sky
class Spec1D():
    def __init__(self,Order,sa=True,write_path=None):
        self.Order   = Order
        self.setting = Order.setting
        self.echelle = Order.echelle
        self.crossdisp = Order.crossdisp
        self.airmass = Order.airmass
        self.onum    = Order.onum
        self.header  = Order.header
        self.Envi    = Order.Envi
        self.sh      = Order.sh[1]
        self.sa      = sa
        self.writepath = write_path

        # Normalized 1D PSF
        PSF = self.getPSF()
        self.disp = self.Envi.getDispersion(self.setting,self.onum)

        # Wavelength guesses for each pixel
        self.wave_pos = self.waveGuess(beam='pos')
        self.wave_neg = self.waveGuess(beam='neg')

        # print ('wave pos & wave neg')
        # print (self.wave_pos)
        # print (self.wave_neg)

        # Get 1D flux for pos and neg orders, as well as sky background
        self.flux_pos,self.uflux_pos,self.flux_neg,self.uflux_neg,self.sky_pos,self.sky_neg,self.usky_pos,self.usky_neg = self.extract(PSF,method='single')

        # print ('flux pos and flux neg')
        # print (self.flux_pos).shape
        # print (self.flux_neg).shape
        #
        # print (self.flux_pos)
        # print (self.flux_neg)

        # ???
        # if sa:
        #     self.sa_pos,self.usa_pos,self.sa_neg,self.usa_neg = self.SpecAst(PSF)

        # if write_path:
        #     self.file = self.writeSpec(path=write_path)

    # Create normalized 1D PSF in a window from x=300 to x=900 pixels
    def getPSF(self,range=(300,900)):
        # PSF is the median of values in the wavelength direction
        PSF = np.median(self.Order.image_rect[:,range[0]:range[1]],1)
        #print (PSF.shape)
        npsf = PSF.size
        # Divide by half the sum of counts (because pos and neg flux each account for half the flux)
        PSF_norm = PSF/PSF[:npsf/2-1].sum()
        #print (PSF_norm.shape)
        return PSF_norm

    # Extract 1D spectra using method from Horne1986
    def extract(self,PSF,method='single'):
        # Placeholders
        pixsig = 1.
        # Frame size
        sh   = self.sh
        npsf = PSF.size

        flux_pos  = np.zeros(sh)
        flux_neg  = np.zeros(sh)

        uflux_pos = np.zeros(sh)
        uflux_neg = np.zeros(sh)

        sky_pos  = np.zeros(sh)
        sky_neg  = np.zeros(sh)        

        usky_pos  = np.zeros(sh)
        usky_neg  = np.zeros(sh)        

        # Load rectified A-B images
        im = self.Order.image_rect
        uim = self.Order.uimage_rect
        sky = self.Order.sky_rect
        usky = self.Order.usky_rect

        # Iterate over wavelength direction
        for i in np.arange(sh):
            #There is something wrong with the optimal weights here. Will have to check on it later. For 
            #now, uniform weights make very little difference for high S/N spectra.

            # Compute fluxes with optimal extraction algorithm (Horne1986)
            flux_pos[i] = (PSF[:npsf/2-1]*im[:npsf/2-1,i]/uim[:npsf/2-1,i]**2).sum() / (PSF[:npsf/2-1]**2/uim[:npsf/2-1,i]**2).sum()
            flux_neg[i] = (PSF[npsf/2:-1]*im[npsf/2:-1,i]/uim[npsf/2:-1,i]**2).sum() / (PSF[npsf/2:-1]**2/uim[npsf/2:-1,i]**2).sum()
            #flux_pos[i] = (PSF[:npsf/2-1]*im[:npsf/2-1,i]).sum() / (PSF[:npsf/2-1]**2).sum()
            #flux_neg[i] = (PSF[npsf/2:-1]*im[npsf/2:-1,i]).sum() / (PSF[npsf/2:-1]**2).sum()

            # Error in fluxes
            uflux_pos[i] = np.sqrt(1.0/(PSF[:npsf/2-1]**2/uim[:npsf/2-1,i]**2.).sum())
            uflux_neg[i] = np.sqrt(1.0/(PSF[npsf/2:-1]**2/uim[npsf/2:-1,i]**2.).sum())

            # Same as above for the sky, but variance is zero (sky assumed to be noise-free)
            sky_pos[i] = (PSF[:npsf/2-1]*sky[:npsf/2-1,i]).sum() / (PSF[:npsf/2-1]**2).sum()
            sky_neg[i] = -(PSF[npsf/2:-1]*sky[npsf/2:-1,i]).sum() / (PSF[npsf/2:-1]**2).sum()
            usky_pos[i] = np.sqrt(1.0/(PSF[:npsf/2-1]**2/usky[:npsf/2-1,i]**2.).sum())
            usky_neg[i] = np.sqrt(1.0/(PSF[npsf/2:-1]**2/usky[npsf/2:-1,i]**2.).sum())

        #print ('PSF and im')
        # print (PSF[:npsf/2-1])
        # print (im[:npsf/2-1,512])
        #
        # print (PSF[npsf / 2:-1])
        # print (im[npsf / 2:-1, 512])

        # Fill in faulty value with 1. for fluxes, 1000 for error, so they will have tiny weight
        flux_pos = ma.masked_invalid(flux_pos)
        flux_pos = ma.filled(flux_pos,1.)
        uflux_pos = ma.masked_invalid(uflux_pos)
        uflux_pos = ma.filled(uflux_pos,1000.)
        sky_pos = ma.masked_invalid(sky_pos)
        sky_pos = ma.filled(sky_pos,1.)
        usky_pos = ma.masked_invalid(usky_pos)
        usky_pos = ma.filled(usky_pos,1000.)
        
        flux_neg = ma.masked_invalid(flux_neg)
        flux_neg = ma.filled(flux_neg,1.)
        uflux_neg = ma.masked_invalid(uflux_neg)
        uflux_neg = ma.filled(uflux_neg,1000.)
        sky_neg = ma.masked_invalid(sky_neg)
        sky_neg = ma.filled(sky_neg,1.)
        usky_neg = ma.masked_invalid(usky_neg)
        usky_neg = ma.filled(usky_neg,1000.)

        # What is this???
        sky_pos_cont = self._fitCont(self.wave_pos,sky_pos)
        sky_neg_cont = self._fitCont(self.wave_neg,sky_neg)

        # print ('flux_pos')
        # print (flux_pos.shape)
        # print ('sky_pos')
        # print (sky_pos.shape)

        time = self.header['UTC']
        time = time.replace(':', '')
        time = time[0:4]
        date = self.header['DATE-OBS']
        date = date.replace('-', '')
        object = self.header['OBJECT']
        object = object.replace(' ', '')

        # Plot positive flux with initial wavelength guess
        plt.plot(self.wave_pos, flux_pos, drawstyle ='steps-mid',label='Pos flux w/ initial wavelength fit')
        plt.plot(self.wave_pos, sky_pos, drawstyle='steps-mid', label='Pos sky w/ initial wavelength fit')
        plt.legend(loc='upper left')
        filename = self.writepath + '/' + object + '_' + date + '_' + time + '_spec1d' + str(self.onum) + '_pos' + '.png'
        print (filename)
        plt.savefig(filename)
        plt.close()

        plt.plot(self.wave_neg, flux_neg, drawstyle='steps-mid', label='Neg flux w/ initial wavelength fit')
        plt.plot(self.wave_neg, sky_neg, drawstyle='steps-mid', label='Neg sky w/ initial wavelength fit')
        plt.legend(loc='upper left')
        filename = self.writepath + '/' + object + '_' + date + '_' + time + '_spec1d' + str(
            self.onum) + '_neg' + '.png'
        print (filename)
        plt.savefig(filename)
        plt.close()

        # Why is sky_pos_cont subtracted??? why is nothing subtracted for flux_pos?
        return flux_pos,uflux_pos,flux_neg,uflux_neg,sky_pos-sky_pos_cont,sky_neg-sky_neg_cont,usky_pos,usky_neg

    #
    def _fitCont(self,wave,spec):
        bg_temp = 210. #K
        
        niter = 2

        cont = self.bb(wave*1e-6,bg_temp)
        #print ('cont')
        #print (cont)
        gsubs = np.where(np.isfinite(spec))
        for i in range(niter):
            norm = np.median(spec[gsubs])
            print ('norm')
            print (norm)
            norm_cont = np.median(cont[gsubs])
            # normalize by specific intensity from a BB continuum?
            cont *= norm/norm_cont
            gsubs = np.where(spec<cont)

        return cont

    # Computes specific intensity: intensity at each wavelength
    def bb(self,wave,T):
        # speed of light
        cc = constants.c
        # Planck constant
        hh = constants.h
        # Boltzmann constant
        kk = constants.k
        
        blambda = 2.*hh*cc**2/(wave**5*(np.exp(hh*cc/(wave*kk*T))-1.))

        return blambda
        

    def SpecAst(self,PSF,method='centroid',width=5):
        '''
        The uncertainty on the centroid is:

                 SUM_j([j*SUM_i(F_i)-SUM_i(i*F_i)]^2 * s(F_j)^2)
        s(C)^2 = ------------------------------------------------
                                [SUM_i(F_i)]^4

        
        '''
        
        #Guesstimated placeholder
        aper_corr = 1.4
        posloc = np.argmax(PSF)
        negloc = np.argmin(PSF)

        sa_pos = np.zeros(self.sh)
        sa_neg = np.zeros(self.sh)

        usa_pos = np.zeros(self.sh)
        usa_neg = np.zeros(self.sh)

        im = self.Order.image_rect
        uim = self.Order.uimage_rect

        for i in np.arange(self.sh):
            index = np.arange(width*2+1)-width

            # First calculate SUM_i(F_i)
            F_pos = (im[posloc-width:posloc+width+1,i]).sum() 
            F_neg = (im[negloc-width:negloc+width+1,i]).sum()
            #print index, negloc, width, i, np.shape(im)
            # then SUM_i(i*F_i)
            iF_pos = (index*im[posloc-width:posloc+width+1,i]).sum()
            iF_neg = (index*im[negloc-width:negloc+width+1,i]).sum()

            sa_pos[i] = iF_pos/F_pos
            sa_neg[i] = iF_neg/F_neg
       
            # Now propagate the error
            uF_pos = uim[posloc-width:posloc+width+1,i]
            uF_neg = uim[negloc-width:negloc+width+1,i]
            usa_pos[i]  = np.sqrt(((index*F_pos - iF_pos)**2 * uF_pos**2).sum())/F_pos**2
            usa_neg[i]  = np.sqrt(((index*F_neg - iF_neg)**2 * uF_neg**2).sum())/F_neg**2

        #NIRSPEC flips the spectrum on the detector (as all echelles do).
        sa_pos[i] = -sa_pos[i]
        sa_neg[i] = -sa_neg[i]

        return sa_pos*aper_corr,usa_pos*aper_corr,sa_neg*aper_corr,usa_neg*aper_corr

    def plot(self):
        plt.plot(self.wave,self.flux_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.flux_neg,drawstyle='steps-mid')
        plt.show()

    def plotSA(self):
        plt.plot(self.wave,self.sa_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.sa_neg,drawstyle='steps-mid')
        plt.show()

    # what are the ABC ranges???
    def waveGuess(self,beam='pos'):
        wrange = self.Envi.getWaveRange(self.setting,self.onum)
        index = np.arange(self.sh)
        if beam is 'pos':
            j = 0
        elif beam is 'neg':
            j = 1
        else:
            raise AttributeError
       
        wave = self.disp['A'][j]+self.disp['B'][j]*index+self.disp['C'][j]*index**2.
        return wave

    def writeSpec(self,filename=None,path='.'):
        
        c1  = pf.Column(name='wave_pos', format='D', array=self.wave_pos)
        c2  = pf.Column(name='flux_pos', format='D', array=self.flux_pos)
        c3  = pf.Column(name='uflux_pos', format='D', array=self.uflux_pos)
        c4  = pf.Column(name='sky_pos', format='D', array=self.sky_pos)
        c5  = pf.Column(name='usky_pos', format='D', array=self.usky_pos)        
        if self.sa:
            c6  = pf.Column(name='sa_pos', format='D', array=self.sa_pos)
            c7  = pf.Column(name='usa_pos', format='D', array=self.usa_pos)
        c8  = pf.Column(name='wave_neg', format='D', array=self.wave_neg)
        c9  = pf.Column(name='flux_neg', format='D', array=self.flux_neg)
        c10  = pf.Column(name='uflux_neg', format='D', array=self.uflux_neg)
        c11  = pf.Column(name='sky_neg', format='D', array=self.sky_neg)
        c12  = pf.Column(name='usky_neg', format='D', array=self.usky_neg)
        if self.sa:
            c13  = pf.Column(name='sa_neg', format='D', array=self.sa_neg)
            c14 = pf.Column(name='usa_neg', format='D', array=self.usa_neg)

        if self.sa:
            coldefs = pf.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14])
        else:
            coldefs = pf.ColDefs([c1,c2,c3,c4,c7,c8,c9,c10,c11,c12])

        #tbhdu = pf.new_table(coldefs)
        tbhdu = pf.BinTableHDU.from_columns(coldefs)

        self.header['SETNAME'] = (self.setting, 'Setting name')
        self.header['ECHLPOS'] = (self.echelle, 'Echelle position')
        self.header['DISPPOS'] = (self.crossdisp, 'Cross disperser position')
        self.header['ORDER'] = (str(self.onum),'Order number')

        self.header['WAVA_POS'] = (self.disp['A'][0],'Dispersion coefficient, positive beam')
        self.header['WAVB_POS'] = (self.disp['B'][0],'Dispersion coefficient, positive beam')
        self.header['WAVC_POS'] = (self.disp['C'][0],'Dispersion coefficient, positive beam')

        self.header['WAVA_NEG'] = (self.disp['A'][1],'Dispersion coefficient, negative beam')
        self.header['WAVB_NEG'] = (self.disp['B'][1],'Dispersion coefficient, negative beam')
        self.header['WAVC_NEG'] = (self.disp['C'][1],'Dispersion coefficient, negative beam')

        self.header['RPOW_POS'] = (self.disp['R'][0],'Resolving power, positive beam')
        self.header['RPOW_NEG'] = (self.disp['R'][1],'Resolving power, negative beam')
        

        hdu = pf.PrimaryHDU(header=self.header)
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            time   = self.header['UTC']
            time   = time.replace(':','')
            time   = time[0:4]
            date   = self.header['DATE-OBS']
            date   = date.replace('-','')
            object = self.header['OBJECT']
            object = object.replace(' ','')
            filename = path+'/'+object+'_'+date+'_'+time+'_spec1d'+str(self.onum)+'.fits'

        
        thdulist.writeto(filename,clobber=True)

        return filename

## Class to calibrate wavelength using rfm
class WaveCal():
    
    def __init__(self,specfile,path='./',am=1., hp=True):
        self.specfile = specfile
        self.path = path
        self.WavePos = rfm.Optimize(specfile,alt=4.145,rpower=29000.,beam='pos',cull=2,am=am)
        self.WaveNeg = rfm.Optimize(specfile,alt=4.145,rpower=29000.,beam='neg',cull=2,am=am) 
        self.file = self._updateWave(specfile,self.WavePos,self.WaveNeg)
        self.plotWaveFit(hp)
        
    def _updateWave(self,specfile,WavePos,WaveNeg):
        spec1d = pf.open(specfile)
        spec1d[1].data.field('wave_pos')[:] = self.WavePos.getWave()
        spec1d[1].data.field('wave_neg')[:] = self.WaveNeg.getWave()
        
        oldpath,filename = os.path.split(specfile)
        length = len(filename)
        onum = filename[length-6]
        filename = filename[0:length-12]
        filename = filename+'wave'+onum+'.fits'
        fullpath = self.path+'/'+filename
        spec1d.writeto(fullpath,clobber=True)
        
        return fullpath
        
    def plotWaveFit(self, hp=True):
        # NRC (19 JUNE 2014) - EDITED plotWaveFit SO THAT THE WAVE CALIBRATION PLOTS DON'T
        # HAVE TO STOP THE PIPELINE FROM RUNNING BY SETTING 'hp=False'.

        plt.close()
        plt.close()

        trans = self.WavePos.getModel()
        plt.figure(self.specfile+' (Pos)')
        plt.plot(trans['wave'],trans['Radiance'],label='radiance')
        plt.plot(trans['wave'],trans['sky'],label='sky')
        plt.legend(loc='upper left')
        if (hp == True):
            plt.show()
        else:
            plt.show(block=False)

        trans = self.WaveNeg.getModel()
        plt.figure(self.specfile+' (Neg)')
        plt.plot(trans['wave'],trans['Radiance'],label='radiance')
        plt.plot(trans['wave'],trans['sky'],label='sky')
        plt.legend(loc='upper left')
        if (hp == True):
            plt.show()
        else:
            plt.show(block=False)

class CalSpec():
    def __init__(self,scifile,stdfile,shift=0.,dtau=0.0,dowave=True,write_path=None,order=0):

        self.scifile = scifile
        self.stdfile = stdfile

        self.Sci,self.header = readNIRSPEC(scifile)
        self.Std,self.header_std = readNIRSPEC(stdfile)

        self._normalize(self.Sci)
        self._normalize(self.Std)
        self._maskLowTrans()
        self._beers(self.Std,dtau)
        self._addFlux(shift)
        self._addSA(shift)

        if write_path:
            self.file = self.writeSpec(path=write_path,order=order)

    def _maskLowTrans(self,thres=0.25):
        bsubs = np.where(self.Std['flux_pos']<thres)
        self.Std.field('flux_pos')[bsubs] = np.nan
        self.Sci.field('flux_pos')[bsubs] = np.nan
        self.Sci.field('sa_pos')[bsubs] = np.nan
        
        bsubs = np.where(self.Std['flux_neg']<thres)
        self.Std.field('flux_neg')[bsubs] = np.nan
        self.Sci.field('flux_neg')[bsubs] = np.nan
        self.Sci.field('sa_neg')[bsubs] = np.nan

    def _specShift(self,spec,shift):
        nx = spec.size
        index = np.arange(nx)
        return np.interp(index-shift,index,spec)
    #
    def _normalize(self,Spec):
        niter = 3
        
        median = 0.
        gsubs = np.where(Spec['flux_pos']>median)
        for i in np.arange(niter):
            median = np.median(Spec['flux_pos'][gsubs])
            gsubs = np.where(Spec['flux_pos']>median)
        Spec.field('flux_pos')[:] /= median
        Spec.field('uflux_pos')[:] /= median

        median = 0.
        gsubs = np.where(Spec['flux_neg']>median)
        for i in np.arange(niter):
            median = np.median(Spec['flux_neg'][gsubs])
            gsubs = np.where(Spec['flux_neg']>median)
        Spec.field('flux_neg')[:] /= median
        Spec.field('uflux_neg')[:] /= median

    def _beers(self,Spec,dtau):
        Spec.field('flux_pos')[:] = np.exp((1.+dtau)*np.log(Spec['flux_pos']))
        Spec.field('flux_neg')[:] = np.exp((1.+dtau)*np.log(Spec['flux_neg']))

    def _addFlux(self,shift):
        
        sci_pos_resamp = np.interp(self.Std['wave_pos'],self.Sci['wave_pos'],self.Sci['flux_pos'])
        sci_neg_resamp = np.interp(self.Std['wave_pos'],self.Sci['wave_neg'],self.Sci['flux_neg'])
        usci_pos_resamp = np.interp(self.Std['wave_pos'],self.Sci['wave_pos'],self.Sci['uflux_pos'])
        usci_neg_resamp = np.interp(self.Std['wave_pos'],self.Sci['wave_neg'],self.Sci['uflux_neg'])

        std_pos_resamp = self.Std['flux_pos']
        std_neg_resamp = np.interp(self.Std['wave_pos'],self.Std['wave_neg'],self.Std['flux_neg'])
        ustd_pos_resamp = self.Std['uflux_pos']
        ustd_neg_resamp = np.interp(self.Std['wave_pos'],self.Std['wave_neg'],self.Std['uflux_neg'])

        self.flux_pos  = sci_pos_resamp/std_pos_resamp
        self.uflux_pos = np.sqrt((usci_pos_resamp/sci_pos_resamp)**2 + (ustd_pos_resamp/std_pos_resamp)**2)
        self.flux_neg  = sci_neg_resamp/std_neg_resamp
        self.uflux_neg = np.sqrt((usci_neg_resamp/sci_neg_resamp)**2 + (ustd_neg_resamp/std_neg_resamp)**2)

        self.flux = (self.flux_pos+self.flux_neg)/2.
        self.uflux = np.sqrt((self.uflux_pos/self.flux_pos)**2+(self.uflux_neg/self.flux_neg)**2)

        self.wave = self.Std['wave_pos']
        
    def _addSA(self,shift):
        self.sa  = (self.Sci['sa_pos']+np.interp(self.wave,self.Sci['wave_neg'],self.Sci['sa_neg']))/2.
        self.usa = np.sqrt(self.Sci['usa_pos']**2+np.interp(self.wave,self.Sci['wave_neg'],self.Sci['usa_neg']**2))/2.
        self.sa  = self._specShift(self.sa,shift)
        self.usa = self._specShift(self.usa,shift)

    def writeSpec(self, filename=None, path='.',order=1):
        c1  = pf.Column(name='wave', format='D', array=self.wave)
        c2  = pf.Column(name='flux', format='D', array=self.flux)
        c3  = pf.Column(name='uflux', format='D', array=self.uflux)
        c4  = pf.Column(name='sa', format='D', array=self.sa)
        c5  = pf.Column(name='usa', format='D', array=self.usa)

        coldefs = pf.ColDefs([c1,c2,c3,c4,c5])

        #tbhdu = pf.new_table(coldefs)
        tbhdu = pf.BinTableHDU.from_columns(coldefs)

        hdu = pf.PrimaryHDU(header=self.header)
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            basename = getBaseName(self.header)
            filename = path+'/'+basename+'_calspec'+str(order)+'.fits'

        thdulist.writeto(filename,clobber=True)

        return filename

    def plotBeams(self):
        plt.plot(self.wave_pos,self.flux_pos)
        plt.plot(self.wave_neg,self.flux_neg)
        plt.show()

    def plotFlux(self):
        plt.plot(self.wave,self.flux,drawstyle='steps-mid')
        plt.show()

    def plotSA(self):
        plt.plot(self.wave,self.sa,drawstyle='steps-mid')
        plt.show()

class SASpec():
    def __init__(self, SA_ortho_file, SA_para_file, write_path=None,order=1):
        self.SA_ortho_file = SA_ortho_file
        self.SA_para_file  = SA_para_file

        self.Spec1,header1 = readNIRSPEC(SA_ortho_file)
        self.Spec2,header2 = readNIRSPEC(SA_para_file)

        self.header = header1
        self._subSA()
        self._addFlux()

        if write_path:
            self.file = self.writeSA(path=write_path,order=order)

    def _subSA(self):
        '''
        Note that we are assuming that the wavescale is close to identical, which
        we anyway need to be the case for SA to work. We could add a check and a warning at this point. 
        '''
        self.wave = (self.Spec1.wave+self.Spec2.wave)/2.
        self.sa = (self.Spec1.sa-self.Spec2.sa)/2.
        self.usa = np.sqrt((self.Spec1.usa**2+self.Spec2.usa**2))/2.

    def _addFlux(self):
        self.flux = (self.Spec1.flux+self.Spec2.flux)/2.
        self.uflux = np.sqrt((self.Spec1.uflux**2+self.Spec2.uflux**2))/2.

    def writeSA(self,filename=None,path='.',order=1):
        c1  = pf.Column(name='wave', format='D', array=self.wave)
        c2  = pf.Column(name='flux', format='D', array=self.flux)
        c3  = pf.Column(name='uflux', format='D', array=self.uflux)
        c4  = pf.Column(name='sa', format='D', array=self.sa)
        c5  = pf.Column(name='usa', format='D', array=self.usa)

        coldefs = pf.ColDefs([c1,c2,c3,c4,c5])

        #tbhdu = pf.new_table(coldefs)
        tbhdu = pf.BinTableHDU.from_columns(coldefs)
        hdu = pf.PrimaryHDU()
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            basename = getBaseName(self.header)
            filename = path+'/'+basename+'_saspec'+str(order)+'.fits'

        thdulist.writeto(filename,clobber=True)

        return filename

# NRC (27 MAY 2014) - EDITED THE Reduction AND SAReduction CLASSES SO THERE IS A 'SettingsFile' ARGUMENT. THE USER
# CAN NOW MORE EASILY CHANGE THE PARAMETERS IN THIS FILE, WHICH DICTATE WHERE THE PIPELINE LOOKS FOR THE ECHELLE 
# ORDERS. THIS IS NECESSARY BECAUSE WHERE THE ORDERS APPEAR ON THE CHIP CAN SHIFT FOR DIFFERENT OBSERVATION EPOCHS,
# AND IF THESE VALUES ARE NOT SET PROPERLY IT CAN CAUSE THE PIPELINE TO FAIL. 

# NRC (19 JUNE 2014) - EDITED THE Reduction AND SAReduction CLASSES SO THAT TARGET AND STANDARD NAMES CAN BE EDITED
# BY THE USER WITH THE sci_tname AND std_tname ARGUMENTS, RESPECTIVELY. THESE NAMES ARE USED IN THE OUTPUT FILES.

# NRC (19 JUNE 2014) - ADDED AN ARGUMENT 'hold_plots' TO THE Reduction AND SAReduction CLASSES SO THAT WHEN SET TO
# True THE PIPELINE WILL NOT BE STOPPED BY THE WAVELENGTH CALIBRATION PLOTS.


class Reduction():
    '''
    Top level basic script for reducing an observation, consisting of a science target and a telluric standard, as well
    as associated calibration files. Output is saved in ../pynirspec-out/ut-date/target-name/
    '''

    # initialize file paths and configuration variables
    def __init__(self,flat_range=None, flat_dark_range=None, dark_range=None,
                 sci_range=None, std_range=None, path=None, base=None,level1=True,level2=True,
                 shift=0.0, dtau=0.0, save_dark=True, save_flat=True, SettingsFile=None,
                 ut_date = None, sci_tname=None, std_tname=None, hold_plots=True, hold=True, **kwargs):

        if (hold == False):
            hold_plots = False

        # Save date and target name for later
        self.ut_date = ut_date
        self.sci_tname = sci_tname

        # Whether to save processed darks and flats
        self.save_dark = save_dark
        self.save_flat = save_flat

        # Range of science images
        sci_range1 = sci_range

        # ???
        self.shift = shift
        self.dtau = dtau

        # Path to save Level1 .json file: a summary of L1 output paths

        date_dir = '../pynirspec-out/'+self.ut_date+'/'
        if not os.path.exists(date_dir):
            os.mkdir(date_dir)
        out_dir = date_dir+self.sci_tname+'/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.level1_path = out_dir+'L1FILES'
        self.spec2d_path = out_dir+'SPEC2D'
        self.spec1d_path = out_dir+'SPEC1D'
        self.wave_path = out_dir+'WAVE'
        self.cal1d_path = out_dir+'CAL1D'
        out_path_list = [self.level1_path, self.spec2d_path, self.spec1d_path, self.wave_path, self.cal1d_path]

        # Create output directories if they do not exist
        for out_path in out_path_list:
            if not os.path.exists(out_path):
                os.mkdir(out_path)

        # Path to .ini file containing instrument settings and details of the spectral orders etc.
        self.SettingsFile = SettingsFile

        # Create lists of file names for each file type
        self.flat_dark_names = makeFilelist(base,flat_dark_range,path=path)
        self.obs_dark_names  = makeFilelist(base,dark_range,path=path)
        self.flat_names = makeFilelist(base,flat_range,path=path)
        self.sci_names1 = makeFilelist(base,sci_range1,path=path)
        self.std_names = makeFilelist(base,std_range,path=path)

        # ???
        self.mode  = 'SciStd'

        # Dict for the data files
        self.tdict = {'science':self.sci_names1} #,'standard':self.std_names}
        # Dict for target names
        self.ndict = {'science':sci_tname}#, 'standard':std_tname}

        # To hold off showing plots in wavelength calibration so that code continues to run
        self.hold_plots = hold_plots

        # Whether to run Level1 steps
        if level1:
            self._level1()

        # Whether to run Level2 steps
        if level2:
            self._level2()

    ## Level 1 processing, includes dark subtraction, bad pixel map, flat-fielding ...
    def _level1(self):

        # Create flat dark
        FDark = Dark(self.flat_dark_names,ut_date=self.ut_date,sci_tname=self.sci_tname)
        # Create observation dark
        ODark = Dark(self.obs_dark_names,save=self.save_dark,ut_date=self.ut_date,sci_tname=self.sci_tname)
        # Create flats
        OFlat = Flat(self.flat_names, dark=FDark,save=self.save_flat, SettingsFile=self.SettingsFile,ut_date=self.ut_date,sci_tname=self.sci_tname)

        img_flat = self.save_flat

        # Initialize dictionary for level1 files
        level1_files = {}
        #print (self.tdict)
        # Iterate over observation types (only science for now)
        for key in self.tdict.keys():
            #print (key)
            ONod    = Nod(self.tdict[key],flat=OFlat,dark=ODark, tname=self.ndict[key], SettingsFile=self.SettingsFile,
                          badpix='../pynirspec-out/'+self.ut_date+'/'+self.sci_tname+'/badpix.dmp')
          
            matrix = ONod.TargetStack

            # Number of spectral orders
            norders = ONod.getNOrders()
            target_files = []
            # Process each order separately
            for i in np.arange(norders):
                print '### Processing order',i+1
                # Shift A-B images, combine them, and rectify combined A-B image
                OOrder   = Order(ONod,onum=i+1,write_path=self.spec2d_path)
                print '### 2D order extracted'
                #
                OSpec1D  = Spec1D(OOrder,sa=True,write_path=self.spec1d_path)

                # Omit -- we will do wavelength calibration differently
                # OWaveCal = WaveCal(OSpec1D.file,path=self.wave_path,am=OSpec1D.airmass, hp=self.hold_plots)
                OOrder_files = {'2d':OOrder.file} #,'1d':OSpec1D.file, 'wave':OWaveCal.file}
                target_files.append(OOrder_files)

            level1_files[key] = target_files

        filename = self._getLevel1File(self.ndict['science']) 
 
        f = open(self.level1_path+'/'+filename, 'w')
        json.dump(level1_files,f)
        f.close()

    def _level2(self):

        filename = self._getLevel1File(self.ndict['science'])
        f = open(self.level1_path+'/'+filename, 'r')
        level1_files = json.load(f)
        f.close()

        norders = len(level1_files['science'])

        for i in np.arange(norders):
            sci_file = level1_files['science'][i]['wave']
            std_file = level1_files['standard'][i]['wave']
            OCalSpec = CalSpec(sci_file,std_file,shift=self.shift,dtau=self.dtau,write_path=self.cal1d_path,order=i+1)


    def _getLevel1File(self, tname):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        header = pf.open(self.sci_names1[0],ignore_missing_end=True)[0].header
        basename = getBaseName(header, tname)
        filename = basename+'_files.json'
        return filename

# class SAReduction(Reduction):
#     #def __init__(self,flat_range=None, flat_dark_range=None, dark_range=None,
#                  #sci_range1=None, sci_range2=None, std_range=None, path=None, base=None,
#                  #level1_path='../pynirspec-out/L1FILES', shift1=0.0, dtau1=0.0, shift2=0.0, dtau2=0.0,
#                  #level1=True, level2=True, save_dark=False, save_flat=False, SettingsFile=None,
#                  #sci_tname=None, std_tname=None, hold_plots=True, hold=True, **kwargs):
#
#     def __init__(self,flat_range=None, flat_dark_range=None, dark_range=None,
#                  sci_range=None, std_range=None, path=None, base=None,
#                  level1_path='../pynirspec-out/L1FILES', shift=0.0, dtau=0.0,
#                  level1=True, level2=True, save_dark=True, save_flat=True, SettingsFile=None,
#                  sci_tname=None, std_tname=None, hold_plots=True, hold=True, **kwargs):
#
#         if (hold == False):
#             hold_plots = False
#
#         self.save_dark = save_dark
#         self.save_flat = save_flat
#
#
#         self.shift = shift
#         self.dtau = dtau
#
#         self.level1_path = level1_path
#
#         self.SettingsFile = SettingsFile
#
#         self.flat_dark_names = makeFilelist(base,flat_dark_range,path=path)
#         self.obs_dark_names  = makeFilelist(base,dark_range,path=path)
#         self.flat_names      = makeFilelist(base,flat_range,path=path)
#         self.sci_names      = makeFilelist(base,sci_range,path=path)
#       #  self.sci_names2      = makeFilelist(base,sci_range2,path=path)
#         self.std_names       = makeFilelist(base,std_range,path=path)
#
#         self.mode  = 'SA'
#         #self.tdict = {'science':self.sci_names1,'anti_science':self.sci_names2,'standard':self.std_names}
#         #self.ndict = {'science':sci_tname, 'anti_science':sci_tname, 'standard':std_tname}
#         self.tdict = {'science':self.sci_names,'standard':self.std_names}
#         self.ndict = {'science':sci_tname,'standard':std_tname}
#         self.hold_plots = hold_plots
#
#         if level1:
#             self._level1()
#
#         if level2:
#             self._level2()
#
#     def _level2(self):
#         filename = self._getLevel1File(self.ndict['science'])
#         f = open(self.level1_path+'/'+filename, 'r')
#         level1_files = json.load(f)
#         f.close()
#
#         norders = len(level1_files['science'])
#
#         for i in np.arange(norders):
#             sci_file  = level1_files['science'][i]['wave']
#             asci_file = level1_files['anti_science'][i]['wave']
#             std_file  = level1_files['standard'][i]['wave']
#             OCalSpec = CalSpec(sci_file,std_file,shift=self.shift1,dtau=self.dtau1,write_path='../pynirspec-out/CAL1D',order=i+1)
#             ACalSpec = CalSpec(asci_file,std_file,shift=self.shift2,dtau=self.dtau2,write_path='../pynirspec-out/CAL1D',order=i+1)
#             OSASpec = SASpec(OCalSpec.file,ACalSpec.file, write_path='../pynirspec-out/SA1D',order=i+1)


def readFilelist(listfile):
    funit = open(listfile)
    flist = funit.readlines()
    return flist

## Create a list of file names for a specific file type (flats, darks etc.)
def makeFilelist(date,ranges,path=''):

    if not isinstance(ranges,list):
        ranges = [ranges]

    fnames = []
    for Range in ranges:
        fnumbers = np.arange(Range[0], Range[1]+1, 1)
        for number in fnumbers:
            fnames.append(path+date+'s'+str(number).zfill(4)+'.fits')

    return fnames

def readNIRSPEC(filename):
    header = pf.open(filename)[0].header
    data = pf.getdata(filename)
    return data, header

def getBaseName(header,tname=None):
    time   = header['UTC']
    time   = time.replace(':','')
    time   = time[0:4]
    date   = header['DATE-OBS']
    date   = date.replace('-','')
    if (tname is None):
        object = header['OBJECT']
    else:
        object = tname
    object = object.replace(' ','')
    basename = object+'_'+date+'_'+time
    return basename

def calc_centroid(cc,cwidth=15):
    maxind = np.argmax(cc)
    
    mini = max([0,maxind-cwidth])
    maxi = min([maxind+cwidth,cc.shape[0]])
    trunc = cc[mini:maxi]
    centroid = mini+(trunc*np.arange(trunc.shape[0])).sum()/trunc.sum()
    
    return centroid

def write_fits(array,filename='test.fits'):
    hdu = pf.PrimaryHDU(array)
    hdu.writeto(filename,clobber=True)
