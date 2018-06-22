import warnings
import json
import os
import ConfigParser as cp

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
user = 'pelletier'


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
        
    def getDetPars(self):
        gain = self.detpars.getfloat('Detector','gain')
        rn   = self.detpars.getfloat('Detector','rn')
        dc   = self.detpars.getfloat('Detector','dc')
        return {'gain':gain,'rn':rn,'dc':dc}
        
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
    type
    Envi
    flist
    planes
    header

    Methods
    -------
    getSetting
    getNOrders
    getTargetName
    subtractFromStack
    divideInStack
    writeImage
   
    '''
    def __init__(self,filelist,type='image', SettingsFile=None,tname=None):
        self.type = type
        self.Envi = Environment(settings_file=SettingsFile)
        self.flist = filelist
        self._openList(tname)
        self._makeHeader(tname)
        
    def _openList(self, tname):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)

        self.planes = []
        for file in self.flist:
            plane = pf.open(file,ignore_missing_end=True)
            self.planes.append(plane)
        self._makeHeader(tname)

        self.exp_pars = self.getExpPars()
        self.det_pars = self.Envi.getDetPars()

    def _makeHeader(self, tname=None):
        self.header = self.planes[0][0].header 
        
        #The NIRSPEC headers have a couple of illegally formatted keywords
        print self.header['FILENAME']          # Added by Bjorn and Stefan
        try:
            del self.header['GAIN.SPE']
        except:
            print ('GAIN.SPE already deleted from hdr.')
        try:
            del self.header['FREQ.SPE']
        except:
            print ('FREQ.SPE already deleted from hdr.')

    def getExpPars(self):
        coadds   = self.getKeyword('COADDS')[0]
        sampmode = self.getKeyword('SAMPMODE')[0]
        nreads   = self.getKeyword('MULTISPE')[0]
        itime    = self.getKeyword('ITIME')[0]
        nexp     = len(self.planes)
        return {'coadds':coadds,'sampmode':sampmode,'nreads':nreads,'itime':itime,'nexp':nexp}

    def getSetting(self):
        echelle   = self.getKeyword('ECHLPOS')
        crossdisp = self.getKeyword('DISPPOS')
# commented out 25 jan 16 by dp bc server crash at flat=399 for 1 dec 2015 data and motors not reintialized, header not written correctly
#        for ii, (e, c,) in enumerate(zip(echelle, crossdisp)) : 
#           print ii, e, c
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

        setsub1 = [i for i,v in enumerate(echelles) if float(v)==echelle]
        # print (setsub1)
        setsub2 = [i for i,v in enumerate(crossdisps) if float(v)==crossdisp]
        # print (setsub2)
        sub = [i for i in setsub1 if i in setsub2]
        # print (sub)
        #print (self.Envi.getSections()[sub[0]])
        return self.Envi.getSections()[sub[0]],echelle,crossdisp

    def getAirmass(self):
        airmasses = self.getKeyword('AIRMASS') 
        airmasses = np.array([airmass if type(airmass) is not str else -1 for airmass in airmasses])
        mean = np.mean(airmasses[np.where(airmasses!=-1)])
        airmasses[np.where(airmasses==-1)] = mean
        return airmasses

    def getNOrders(self):
        setting,echelle,crossdisp = self.getSetting()
        return self.Envi.getNOrders(setting)

    def getTargetName(self):
        target_name = self.header['OBJECT']
        return target_name

    def _getStack(self):
        
        nexp = self.exp_pars['nexp']
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']
        
        stack = np.zeros((nx,ny,nexp))
        ustack = np.zeros((nx,ny,nexp))
        
        for i,plane in enumerate(self.planes):
            stack[:,:,i]  = plane[0].data*self.det_pars['gain'] #convert everything to e-
            ustack[:,:,i] = self._error(plane[0].data)
        return stack,ustack

    def _error(self,data):
        var_data = np.abs(data+self.exp_pars['itime']*self.det_pars['dc']+
                          self.det_pars['rn']**2/self.exp_pars['nreads'])
        return np.sqrt(var_data)
    
    def subtractFromStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
        try:
            self.uimage = np.sqrt(self.uimage**2+Obs.uimage**2)
            self.image -= Obs.image
        except:
            print 'Subtraction failed - no image calculated'

    def divideInStack(self,Obs):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

        for i in np.arange(self.stack.shape[2]):
            plane = self.stack[:,:,i]
            uplane = self.ustack[:,:,i]
            uplane = np.sqrt((uplane/plane)**2+(Obs.uimage/Obs.image)**2)
            plane /= Obs.image
            uplane *= np.abs(plane)

            self.stack[:,:,i]  = plane
            self.ustack[:,:,i] = uplane
    
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
        
        masked_stack = ma.masked_invalid(stack)
        masked_ustack = ma.masked_invalid(ustack)
        
        # print (masked_stack.shape)
        # print (masked_ustack.shape)

        image = ma.average(masked_stack,2,weights=1./masked_ustack**2)
        uimage = np.sqrt(ma.mean(masked_ustack**2,2)/ma.count(masked_ustack,2))

        # print (image.shape)
        # print (uimage.shape)

        return image, uimage

    def writeImage(self,filename=None):
        if filename is None:
            filename = self.type+'.fits'
            
        hdu = pf.PrimaryHDU(self.image.data)
        uhdu = pf.ImageHDU(self.uimage.data)
        hdulist = pf.HDUList([hdu,uhdu])
        hdulist.writeto(filename,clobber=True)

    def getKeyword(self,keyword):
        try:
            klist = [plane[0].header[keyword] for plane in self.planes]
            return klist
        except ValueError:
            print "Invalid header keyword"
        
class Flat(Observation):
    def __init__(self,filelist,dark=None,norm_thres=5000.,save=False,**kwargs):
        
        Observation.__init__(self,filelist,**kwargs)
        self.type = 'flat'
        self.stack,self.ustack = self._getStack()
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]
        
        self.setting,self.echelle,self.crossdisp = self.getSetting()

        self.image,self.uimage = self._collapseStack()

        if dark:
            self.subtractFromStack(dark)

        self._normalize(norm_thres)

        #Where the flat field is undefined, it's set to 1 to avoid divide by zeros.
        self.image[np.where(self.image<0.1)] = 1
        if save:
            self.writeImage()
        plt.figure()
        plt.imshow(self.image)
        #plt.close()        
        plt.title('Flats')
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/flats.pdf')
        plt.close()
        
    def _normalize(self,norm_thres):
        flux = np.median(self.image[np.where(self.image>norm_thres)])
        self.image = self.image/flux
        self.uimage = self.uimage/flux

    def makeMask(self):
        return np.where(self.image<0.1,0,1)

class Dark(Observation):
    def __init__(self,filelist,save=False,**kwargs):

        Observation.__init__(self,filelist, **kwargs)
        self.type = 'dark'
        self.stack,self.ustack = self._getStack()
        self.nplanes = self.stack.shape[2]
        self.height = self.stack[:,:,0].shape[0]

        self.image,self.uimage = self._collapseStack()
        self._badPixMap()
        if save:
            self.writeImage()
        plt.figure()
        plt.imshow(self.image)
        #plt.close()	       
        plt.title('Darks')
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/darks.pdf')
        plt.close()
        
    def _badPixMap(self,clip=30,filename='badpix.dmp'):
        median = np.median(self.image)
        var  = tvar(self.image,(-100,100))
        self.badpix = ma.masked_greater(self.image-median,clip*np.sqrt(var))
        if filename is not None:
            self.badpix.dump(filename)
        
class Nod(Observation):
    def __init__(self,filelist,dark=None,flat=None,badpix='badpix.dmp',**kwargs):

        Observation.__init__(self,filelist,**kwargs)
        self.type = 'nod'
                
        self.setting,self.echelle,self.crossdisp = self.getSetting()
        self.airmasses = self.getAirmass()

        self.airmass = np.mean(self.airmasses)
        
        RAs  = self.getKeyword('RA')
        DECs = self.getKeyword('DEC')
        FileNums = self.getKeyword('FILENUM')        

        #A average nod requires a Pair Stack. 
        pairs = self._getPairs(RAs,DECs,FileNums)
        self.stack,self.ustack = self._makePairStack(pairs)

        self.height = self.stack[:,:,0].shape[0]
        if flat:
            self.divideInStack(flat)
        if badpix:
            badmask = np.load(badpix)
            self._correctBadPix(badmask)

        self.TargetStack, self.UTargetStack = self.stack, self.ustack
        
        #An averaged sky frame is constructed using the off-beam pixels
        stackA,ustackA,stackB,ustackB = self._makeSingleBeamStacks(pairs)
        beamStacks = [(stackA,ustackA),(stackB,ustackB)] 

        beam_sky_stacks, beam_usky_stacks = [], []
        for beamStack in beamStacks:
            self.stack = beamStack[0] 
            self.ustack = beamStack[1]   
            if flat:
                self.divideInStack(flat)
            if badpix:
                badmask = np.load(badpix)
                self._correctBadPix(badmask)
            
            beam_sky_stacks.append(self.stack)
            beam_usky_stacks.append(self.ustack)

        self.beamSkyStacks  = beam_sky_stacks
        self.beamUSkyStacks = beam_usky_stacks

            
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
            

    def _makePairStack(self,pairs):
        npairs = len(pairs)
        nx = self.planes[0][0].header['NAXIS1']
        ny = self.planes[0][0].header['NAXIS2']

        stack,ustack = self._getStack()
        pair_stack  = np.zeros((nx,ny,npairs))
        pair_ustack = np.zeros((nx,ny,npairs))
        
        for i,pair in enumerate(pairs):
            pair_stack[:,:,i] = stack[:,:,pair[0]] - stack[:,:,pair[1]]
            pair_ustack[:,:,i] = np.sqrt(ustack[:,:,pair[0]]**2 + ustack[:,:,pair[1]]**2)
            
        return pair_stack,pair_ustack

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

        return pairs


class Order():
    def __init__(self,Nod,onum=1,trace=None,write_path=None):
        self.type = 'order'
        self.header  = Nod.header
        self.setting = Nod.setting
        self.echelle = Nod.echelle
        self.crossdisp = Nod.crossdisp
        self.airmass = Nod.airmass
        
        self.Envi    = Nod.Envi
        self.onum    = onum

        self.yrange = self.Envi.getYRange(self.setting,onum)
        
        #COMBINE IMAGES FOR SOURCE EXTRACTION
        self.stack  = Nod.TargetStack[self.yrange[0]:self.yrange[1],:,:]
#        print np.shape(self.stack)
#        for ind,val in enumerate(self.stack[0,0,:]) : 
#            plt.figure()
#            plt.imshow(self.stack[:,:,ind])
#            plt.savefig('/home/dpiskorz/Desktop/KBandData/decstack'+str(ind)+'.pdf')
        self.ustack = Nod.UTargetStack[self.yrange[0]:self.yrange[1],:,:]
        nexp = len(self.stack[0,0,:])

        offsets1, offsets2 = self._findYOffsets()
        self.stack1  = self._yShift(offsets1,self.stack)
        self.ustack1 = self._yShift(offsets1,self.ustack)
        self.stack2  = self._yShift(offsets2,self.stack)
        self.ustack2 = self._yShift(offsets2,self.ustack)

        self.image1, self.uimage1 = self._collapseOrder(stack=self.stack1, ustack=self.ustack1)
        self.image2, self.uimage2 = self._collapseOrder(stack=self.stack2, ustack=self.ustack2)
        plt.figure()
        plt.imshow(self.image1)
        plt.savefig('/home/'+ user +'/Desktop/LBandData/spec1.pdf')
        plt.close()
        plt.figure()
        plt.imshow(self.image2)
        plt.savefig('/home/'+ user +'/Desktop/LBandData/spec2')
        plt.close()
        if trace is None:
            yr1, trace1 = self.fitTrace(self.image1, 1)
            yr2, trace2 = self.fitTrace(self.image2, 2)
            yrs, traces  = [yr1, yr2], [trace1, trace2]
            
        images, uimages = [self.image1, self.image2], [self.uimage1, self.uimage2]

        self.image_rect, self.uimage_rect = self.yRectify(images, uimages, yrs, traces)
        self.sh = self.image_rect.shape

        self._subMedian()

        #COMBINE SKY IMAGES
        self.beamSkyStacks  = Nod.beamSkyStacks  
        self.beamUSkyStacks = Nod.beamUSkyStacks

        beamSkyRect, beamUSkyRect = [], []
        for beamSkyStack, beamUSkyStack in zip(self.beamSkyStacks, self.beamUSkyStacks):
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
        masked_stack = ma.masked_invalid(stack)
        masked_ustack = ma.masked_invalid(ustack)
        
        image = ma.average(masked_stack,2,weights=1./masked_ustack**2)
        uimage = np.sqrt(ma.mean(masked_ustack**2,2)/ma.count(masked_ustack,2))
        
        return image, uimage


    def _yShift(self,offsets,stack):

        sh = stack.shape
        internal_stack = np.zeros(sh)

        index = np.arange(sh[0])
        for plane in np.arange(sh[2]):
            for i in np.arange(sh[1]):
                col = np.interp(index-offsets[plane],index,stack[:,i,plane])
                internal_stack[:,i,plane] = col

        return internal_stack


    def _findYOffsets(self, kwidth=50):
        sh = self.stack.shape
        yr1 = (0, sh[0]/2-1)     #Bottom half of order (A or B pos)
        yr2 = (sh[0]/2,sh[0]-1)  #Top half of order (other A or B pos)
        yrs = [yr1,yr2]

        offsets1 = np.empty(0)
        offsets2 = np.empty(0) 
        for i in range(0, len(yrs)):
            yr = yrs[i]
            yindex = np.arange(yr[0],yr[1]) #Top or Bottom of Order
            kernel_image = self.stack[:,:,0]
            kernel_o = np.median(kernel_image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth],1)
            kernel_med = np.median(kernel_o)
            kernel = np.subtract(kernel_o, kernel_med)

            for j in range(0,sh[2]):
                image = self.stack[:,:,j]                
                profile_o = np.median(image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth],1)
                profile_med = np.median(profile_o)
                profile = np.subtract(profile_o, profile_med)
                cc = fp.ifft(fp.fft(kernel)*np.conj(fp.fft(profile)))
                cc_sh = fp.fftshift(cc)
                cen = calc_centroid(cc_sh).real - yindex.shape[0]/2.

                ## PLOT ##
                #fig = plt.figure('Order ' + str(self.onum) + ' / AB set ' + str(j+1))
                #ax = fig.add_subplot(111)
                #if (j == 0):
                #    kernel_plot = np.interp(yindex-cen,yindex,profile)
                #ax.plot(yindex, kernel_plot, color='blue')
                #ax.plot(yindex, profile, color='green')
                #profile_new = np.interp(yindex-cen,yindex,profile)
                #ax.plot(yindex, profile_new, color='red')
                #plt.show()
                #####

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


    def fitTrace(self, image, OneOrTwo, kwidth=10):
        sh = image.shape
        #yr1 = (0,sh[0]/2-1) #Bottom half of order (A or B pos)
        #yr2 = (sh[0]/2,sh[0]-1) #Top half of order (other A or B pos)
        #yrs = [yr1,yr2]
       

        if (OneOrTwo == 1):
            yr = (0,sh[0]/2-1)
        if (OneOrTwo == 2):
            yr = (sh[0]/2,sh[0]-1)
        
        yindex = np.arange(yr[0],yr[1]) #Top or Bottom of Order
        kernel = np.median(image[yindex,sh[1]/2-kwidth:sh[1]/2+kwidth],1)
        
 
        centroids = []
        totals = []
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
        median_totals = np.nanmedian(totals)            
        
        xindex = np.arange(sh[1])
        gsubs = np.where((np.isnan(centroids)==False) & (totals>median_totals*0.25) & (totals<median_totals*1.75))


        centroids[gsubs] = median_filter(centroids[gsubs],size=50)
        coeffs = np.polyfit(xindex[gsubs],centroids[gsubs],3)

        poly = np.poly1d(coeffs)
            
        return yr,poly

    def yRectify(self,images,uimages,yrs,traces): 

        sh = images[0].shape
        image_rect = np.zeros(sh)
        uimage_rect = np.zeros(sh)
        
        for yr,trace,image,uimage in zip(yrs,traces,images,uimages):
            index = np.arange(yr[0],yr[1])
            for i in np.arange(sh[1]):
                col = np.interp(index-trace(i),index,image[index,i])
                image_rect[index,i] = col
                col = np.interp(index-trace(i),index,uimage[index,i])
                uimage_rect[index,i] = col

        return image_rect,uimage_rect
        
        
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

        hdulist.writeto(filename,overwrite=True)

        return filename

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


        PSF = self.getPSF()

        self.disp = self.Envi.getDispersion(self.setting,self.onum)
        self.wave_pos = self.waveGuess(beam='pos')
        self.wave_neg = self.waveGuess(beam='neg')
        
        self.flux_pos,self.uflux_pos,self.flux_neg,self.uflux_neg,self.sky_pos,self.sky_neg,self.usky_pos,self.usky_neg = self.extract(PSF,method='single')
        if sa:
            self.sa_pos,self.usa_pos,self.sa_neg,self.usa_neg = self.SpecAst(PSF)

        if write_path:
            self.file = self.writeSpec(path=write_path)

    def getPSF(self,range=(300,700)):
        PSF = np.median(self.Order.image_rect[:,range[0]:range[1]],1) # cut from channel 300 to channel 700 (DP 19 JAN 2016)
        npsf = PSF.size
        PSF_norm = PSF/PSF[:npsf/2-1].sum()
        plt.figure()
        plt.plot(PSF_norm)
        plt.savefig('/home/'+ user +'/Desktop/LBandData/psf_{}_wave{}.pdf'.format(self.header['DATE-OBS'],self.Order))
        plt.plot()
        plt.close()
        #plt.show(block=False)
        return PSF_norm
        
    def extract(self,PSF,method='single'):
        #Placeholder
        pixsig = 1.
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
        
        im = self.Order.image_rect
        uim = self.Order.uimage_rect
        sky = self.Order.sky_rect
        usky = self.Order.usky_rect
        plt.figure()
        plt.imshow(self.Order.image_rect)
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/decimage_rect.pdf')
        plt.close()
        plt.figure()
        plt.imshow(self.Order.uimage_rect)
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/decuimage_rect.pdf')
        plt.close()
        plt.figure()
        plt.imshow(self.Order.sky_rect)
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/decsky_rect.pdf')
        plt.close()
        plt.figure()
        plt.imshow(self.Order.usky_rect)
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/decusky_rect.pdf')
        plt.close()
        plt.figure()
        plt.plot(PSF)
        #plt.show()
        plt.savefig('/home/'+ user +'/Desktop/LBandData/decPSF.pdf')
        plt.close()
#        f, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, sharex=True, sharey=False)
#        ax0.plot(im[46,:])
#        ax1.plot(im[47,:])
#        ax2.plot(im[48,:])
#        ax3.plot(im[49,:])
#        ax4.plot(im[50,:])
#        ax5.plot(im[51,:])
#        f.savefig('/home/dpiskorz/Desktop/KBandData/decim_trace.pdf')
#        f, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, sharex=True, sharey=False)
#        ax0.plot(uim[46,:])
#        ax1.plot(uim[47,:])
#        ax2.plot(uim[48,:])
#        ax3.plot(uim[49,:])
#        ax4.plot(uim[50,:])
#        ax5.plot(uim[51,:])
#        f.savefig('/home/dpiskorz/Desktop/KBandData/decuim_trace.pdf')
        for i in np.arange(sh):
            #There is something wrong with the optimal weights here. Will have to check on it later. For 
            #now, uniform weights make very little difference for high S/N spectra. 
            flux_pos[i] = (PSF[:npsf/2-1]*im[:npsf/2-1,i]/uim[:npsf/2-1,i]**2).sum() / (PSF[:npsf/2-1]**2/uim[:npsf/2-1,i]**2).sum()
            flux_neg[i] = (PSF[npsf/2:-1]*im[npsf/2:-1,i]/uim[npsf/2:-1,i]**2).sum() / (PSF[npsf/2:-1]**2/uim[npsf/2:-1,i]**2).sum()
#            if i == 20 : print PSF[:npsf/2-1]
#            if i == 20 : print im[:npsf/2-1,i]
#            if i == 20 : print uim[npsf/2:-1,i]
#            if i == 20 : print (PSF[npsf/2:-1]*im[npsf/2:-1,i]/uim[npsf/2:-1,i]**2).sum()
#            if i == 20 : print (PSF[npsf/2:-1]**2/uim[npsf/2:-1,i]**2).sum()
#            if i == 20 : print flux_pos[i]
            #flux_pos[i] = (PSF[:npsf/2-1]*im[:npsf/2-1,i]).sum() / (PSF[:npsf/2-1]**2).sum()
            #flux_neg[i] = (PSF[npsf/2:-1]*im[npsf/2:-1,i]).sum() / (PSF[npsf/2:-1]**2).sum()
            uflux_pos[i] = np.sqrt(1.0/(PSF[:npsf/2-1]**2/uim[:npsf/2-1,i]**2.).sum())
            uflux_neg[i] = np.sqrt(1.0/(PSF[npsf/2:-1]**2/uim[npsf/2:-1,i]**2.).sum())
            
            sky_pos[i] = (PSF[:npsf/2-1]*sky[:npsf/2-1,i]).sum() / (PSF[:npsf/2-1]**2).sum()
            sky_neg[i] = -(PSF[npsf/2:-1]*sky[npsf/2:-1,i]).sum() / (PSF[npsf/2:-1]**2).sum()
            usky_pos[i] = np.sqrt(1.0/(PSF[:npsf/2-1]**2/usky[:npsf/2-1,i]**2.).sum())
            usky_neg[i] = np.sqrt(1.0/(PSF[npsf/2:-1]**2/usky[npsf/2:-1,i]**2.).sum())

        plt.figure()
        plt.plot(flux_pos)
        plt.savefig('/home/'+ user +'/Desktop/LBandData/flux_pos.pdf')
        plt.close()
        plt.figure()
        plt.plot(flux_neg)
        plt.savefig('/home/'+ user +'/Desktop/LBandData/flux_neg.pdf')
        plt.close()
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
        
#        plt.figure()
#        plt.plot(flux_pos)
#        plt.savefig('/home/dpiskorz/Desktop/KBandData/flux_pos_masked.pdf')
#        plt.figure()
#        plt.plot(flux_neg)
#        plt.savefig('/home/dpiskorz/Desktop/KBandData/flux_neg_masked.pdf')
        
        sky_pos_cont = self._fitCont(self.wave_pos,sky_pos)
        sky_neg_cont = self._fitCont(self.wave_neg,sky_neg)
        return flux_pos,uflux_pos,flux_neg,uflux_neg,sky_pos-sky_pos_cont,sky_neg-sky_neg_cont,usky_pos,usky_neg
        
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
            cont *= norm/norm_cont 
            gsubs = np.where(spec<cont)

        return cont
        
    def bb(self,wave,T):
        cc = constants.c
        hh = constants.h
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
        
        print "posloc", posloc
        print "negloc", negloc
        print np.shape(im)

        for i in np.arange(self.sh):
            index = np.arange(width*2+1)-width

            # First calculate SUM_i(F_i)
            #print "posloc", posloc,width
            #print type(im)
            #print np.shape(im)
            #print i
            F_pos = (im[posloc-width:posloc+width+1,i]).sum() 
            #print "negloc", negloc,width
            F_neg = (im[negloc-width:negloc+width+1,i]).sum()
            #print index, negloc, width, i, np.shape(im)
            #print im[posloc-width:posloc+width+1,i]
            #print im[negloc-width:negloc+width+1,i]
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
        
#        plt.figure()
#        plt.plot(index)
#        plt.savefig('/home/dpiskorz/Desktop/KBandData/sa_index.pdf')
#        plt.figure()
#        plt.plot(F_pos)
#        plt.savefig('/home/dpiskorz/Desktop/KBandData/sa_F_pos.pdf')
#        plt.figure()
#        plt.plot(iF_pos)
#        plt.savefig('/home/dpiskorz/Desktop/KBandData/sa_if_pos.pdf')
#        plt.figure()
#        plt.plot(sa_pos)
#        plt.savefig('/home/dpiskorz/Desktop/KBandData/sa_pos.pdf')

        return sa_pos*aper_corr,usa_pos*aper_corr,sa_neg*aper_corr,usa_neg*aper_corr

    def plot(self):        
        plt.plot(self.wave,self.flux_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.flux_neg,drawstyle='steps-mid')
        #plt.show()

    def plotSA(self):
        plt.plot(self.wave,self.sa_pos,drawstyle='steps-mid')
        plt.plot(self.wave,self.sa_neg,drawstyle='steps-mid')
        #plt.show()

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

        
        thdulist.writeto(filename,overwrite=True)

        return filename

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
        spec1d.writeto(fullpath,overwrite=True)
        
        return fullpath
        
    def plotWaveFit(self, hp=True):
        # NRC (19 JUNE 2014) - EDITED plotWaveFit SO THAT THE WAVE CALIBRATION PLOTS DON'T
        # HAVE TO STOP THE PIPELINE FROM RUNNING BY SETTING 'hp=False'.

        plt.close()
        plt.close()

        trans = self.WavePos.getModel()
        '''
        plt.figure(self.specfile+' (Pos)')
        plt.plot(trans['wave'],trans['Radiance'])
        plt.plot(trans['wave'],trans['sky'])
        if (hp == True):
            plt.show()
        else:
            plt.show(block=False)
		'''
		
        trans = self.WaveNeg.getModel()
        '''
        plt.figure(self.specfile+' (Neg)')
        plt.plot(trans['wave'],trans['Radiance'])
        plt.plot(trans['wave'],trans['sky'])
        if (hp == True):
            plt.show()
        else:
            plt.show(block=False)
		'''

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

        tbhdu = pf.BinTableHDU.from_columns(coldefs)
        hdu = pf.PrimaryHDU(header=self.header)
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            basename = getBaseName(self.header)
            filename = path+'/'+basename+'_calspec'+str(order)+'.fits'

        thdulist.writeto(filename,overwrite=True)

        return filename

    def plotBeams(self):
        plt.plot(self.wave_pos,self.flux_pos)
        plt.plot(self.wave_neg,self.flux_neg)
        #plt.show()

    def plotFlux(self):
        plt.plot(self.wave,self.flux,drawstyle='steps-mid')
        #plt.show()

    def plotSA(self):
        plt.plot(self.wave,self.sa,drawstyle='steps-mid')
        #plt.show()

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

        tbhdu = pf.BinTableHDU.from_columns(coldefs)
        hdu = pf.PrimaryHDU()
        thdulist = pf.HDUList([hdu,tbhdu])

        if filename is None:
            basename = getBaseName(self.header)
            filename = path+'/'+basename+'_saspec'+str(order)+'.fits'

        thdulist.writeto(filename,overwrite=True)

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
    as associated calibration files. 
    '''
    def __init__(self,flat_range=None, flat_dark_range=None, dark_range=None,
                 sci_range=None, std_range=None, path=None, base=None,level1=True,level2=True,
                 level1_path='L1FILES',shift=0.0, dtau=0.0, save_dark=False, save_flat=False, SettingsFile=None,
                 sci_tname=None, std_tname=None, hold_plots=True, hold=True, **kwargs):

        if (hold == False):
            hold_plots = False

        self.save_dark = save_dark
        self.save_flat = save_flat

        sci_range1 = sci_range
        self.shift       = shift
        self.dtau        = dtau
        self.level1_path = level1_path

        self.SettingsFile = SettingsFile

        self.flat_dark_names = makeFilelist(base,flat_dark_range,path=path)
        self.obs_dark_names  = makeFilelist(base,dark_range,path=path)
        self.flat_names      = makeFilelist(base,flat_range,path=path)
        self.sci_names1      = makeFilelist(base,sci_range1,path=path)
        self.std_names       = makeFilelist(base,std_range,path=path)

        self.mode  = 'SciStd'
        self.tdict = {'science':self.sci_names1} #,'standard':self.std_names}
        self.ndict = {'science':sci_tname}#, 'standard':std_tname}
        self.hold_plots = hold_plots

        if level1:
            self._level1()

        if level2:
            self._level2()

    def _level1(self):
		FDark = Dark(self.flat_dark_names)
		ODark = Dark(self.obs_dark_names,save=self.save_dark)
		OFlat = Flat(self.flat_names, dark=FDark,save=self.save_flat, SettingsFile=self.SettingsFile)
        
		level1_files = {}
		for key in self.tdict.keys():
			ONod    = Nod(self.tdict[key],flat=OFlat,dark=ODark, tname=self.ndict[key], SettingsFile=self.SettingsFile) 
			norders = ONod.getNOrders()
			target_files = []
			for i in np.arange(norders):
				#if i == 0: continue					### if i == 0: continue, skips 0
				#if i == 1: continue
				#if i == 4: continue
				#if i == 2: continue
				print '### Processing order', i+1
				OOrder   = Order(ONod,onum=i+1,write_path='SPEC2D')
				print OOrder.fitTrace(OOrder.image1, 1)
				#plt.figure()
				#plt.imshow(OOrder.uimage1, hold=True)
				#plt.show()
				#plt.figure()
				#plt.imshow(OOrder.uimage_rect, hold=True)
				#plt.show()
				print '### 2D order extracted'
				OSpec1D  = Spec1D(OOrder,sa=True,write_path='SPEC1D')
				OWaveCal = WaveCal(OSpec1D.file,path='WAVE',am=OSpec1D.airmass, hp=self.hold_plots)
				OOrder_files = {'2d':OOrder.file, '1d':OSpec1D.file, 'wave':OWaveCal.file}
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
            OCalSpec = CalSpec(sci_file,std_file,shift=self.shift,dtau=self.dtau,write_path='CAL1D',order=i+1)


    def _getLevel1File(self, tname):
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        header = pf.open(self.sci_names1[0],ignore_missing_end=True)[0].header
        basename = getBaseName(header, tname)
        filename = basename+'_files.json'
        return filename

class SAReduction(Reduction):
    def __init__(self,flat_range=None, flat_dark_range=None, dark_range=None,
                 sci_range1=None, sci_range2=None, std_range=None, path=None, base=None,
                 level1_path='L1FILES', shift1=0.0, dtau1=0.0, shift2=0.0, dtau2=0.0,
                 level1=True, level2=True, save_dark=False, save_flat=False, SettingsFile=None,
                 sci_tname=None, std_tname=None, hold_plots=True, hold=True, **kwargs):

        if (hold == False):
            hold_plots = False
        
        self.save_dark = save_dark
        self.save_flat = save_flat
        
        self.shift1 = shift1
        self.dtau1  = dtau1
        self.shift2 = shift2
        self.dtau2  = dtau2
        
        self.level1_path = level1_path
        
        self.SettingsFile = SettingsFile

        self.flat_dark_names = makeFilelist(base,flat_dark_range,path=path)
        self.obs_dark_names  = makeFilelist(base,dark_range,path=path)
        self.flat_names      = makeFilelist(base,flat_range,path=path)
        self.sci_names1      = makeFilelist(base,sci_range1,path=path)
        self.sci_names2      = makeFilelist(base,sci_range2,path=path)
        self.std_names       = makeFilelist(base,std_range,path=path)

        self.mode  = 'SA'
        self.tdict = {'science':self.sci_names1,'anti_science':self.sci_names2,'standard':self.std_names}
        self.ndict = {'science':sci_tname, 'anti_science':sci_tname, 'standard':std_tname}
        self.hold_plots = hold_plots

        if level1:
            self._level1()
        
        if level2:
            self._level2()

    def _level2(self):   
        filename = self._getLevel1File(self.ndict['science'])
        f = open(self.level1_path+'/'+filename, 'r')
        level1_files = json.load(f)
        f.close()

        norders = len(level1_files['science'])

        for i in np.arange(norders):
            sci_file  = level1_files['science'][i]['wave']
            asci_file = level1_files['anti_science'][i]['wave']
            std_file  = level1_files['standard'][i]['wave']
            OCalSpec = CalSpec(sci_file,std_file,shift=self.shift1,dtau=self.dtau1,write_path='CAL1D',order=i+1)
            ACalSpec = CalSpec(asci_file,std_file,shift=self.shift2,dtau=self.dtau2,write_path='CAL1D',order=i+1)
            OSASpec = SASpec(OCalSpec.file,ACalSpec.file, write_path='SA1D',order=i+1)

def readFilelist(listfile):
    funit = open(lisfile)
    flist = funit.readlines()
    return flist

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
    hdu.writeto(filename,overwrite=True)
