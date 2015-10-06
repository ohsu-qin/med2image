# System imports
import     os
import     sys
import     argparse
import     time
import     glob
import     numpy             as         np
from       random            import     randint
import     re

# System dependency imports
import     nibabel           as         nib
import     dicom
import     pylab
import     matplotlib.cm     as         cm

# Project specific imports
from       .                 import     error
from       .                 import     message        as msg
from       ._common          import     systemMisc     as misc
from       ._common._colors  import     Colors


class ArgumentError(Exception):
    '''Method parameter is missing or invalid.'''
    pass

def run(inputFile, **kwargs):
    '''
    Convert the image as specified by the given arguments.
    
    :param inputFile: the location of the image file to convert
    :param kwargs: the keyword options
    :return: the converted output file location
    '''
    str_outputFileStem, str_outputFileExtension     = os.path.splitext(args.outputFileStem)
    if str_outputFileExtension:
        str_outputFileExtension = str_outputFileExtension.split('.')[1]
    str_inputFileStem,  str_inputFileExtension      = os.path.splitext(args.inputFile)
    
    if not args.outputFileType and str_outputFileExtension:
        args.outputFileType = str_outputFileExtension

    if str_outputFileExtension:
        args.outputFileStem = str_outputFileStem

    if str_inputFileExtension in ['.nii', '.gz']:
        C_convert     = med2image_nii(
                                inputFile         = args.inputFile,
                                outputDir         = args.outputDir,
                                outputFileStem    = args.outputFileStem,
                                outputFileType    = args.outputFileType,
                                sliceToConvert    = args.sliceToConvert,
                                frameToConvert    = args.frameToConvert,
                                showSlices        = args.showSlices,
                                reslice           = args.reslice
                            )
    elif str_inputFileExtension == '.dcm':
        C_convert   = med2image_dcm(
                                inputFile         = args.inputFile,
                                outputDir         = args.outputDir,
                                outputFileStem    = args.outputFileStem,
                                outputFileType    = args.outputFileType,
                                sliceToConvert    = args.sliceToConvert,
                                reslice           = args.reslice
                             )
    else:
        raise NotImplementedError("The extension %s is not supported" %
                                  str_inputFileExtension)

    # And now run it!
    C_convert.run()
    

class med2image(object):
    '''
        med2image accepts as input certain medical image formatted data
        and converts each (or specified) slice of this data to a graphical
        display format such as png or jpg.

    '''
        
    def log(self, *args):
        '''
        get/set the internal pipeline log message object.

        Caller can further manipulate the log object with object-specific
        calls.
        '''
        if args:
            self._log = args[0]
        else:
            return self._log

    def name(self, *args):
        '''
        get/set the descriptive name text of this object.
        '''
        if args:
            self.__name = args[0]
        else:
            return self.__name

    def description(self, *args):
        '''
        Get / set internal object description.
        '''
        if args:
            self._str_desc = args[0]
        else:
            return self._str_desc

    def log(self): return self._log

    @staticmethod
    def urlify(astr, astr_join = '_'):
        # Remove all non-word characters (everything except numbers and letters)
        astr = re.sub(r"[^\w\s]", '', astr)
        
        # Replace all runs of whitespace with an underscore
        astr = re.sub(r"\s+", astr_join, astr)
        
        return astr

    def __init__(self, **kwargs):

        #
        # Object desc block
        #
        self._str_desc                  = ''
        self._log                       = msg.Message()
        self._log._b_syslog             = True
        self.__name                     = "med2image"

        # Directory and filenames
        self._str_workingDir            = ''
        self._str_inputFile             = ''
        self._str_outputFileStem        = ''
        self._str_outputFileType        = ''
        self._str_outputDir             = ''
        self._str_inputDir              = ''

        self._b_convertAllSlices        = False
        self._str_sliceToConvert        = ''
        self._str_frameToConvert        = ''
        self._sliceToConvert            = -1
        self._frameToConvert            = -1

        self._str_stdout                = ""
        self._str_stderr                = ""
        self._exitCode                  = 0

        # The actual data volume and slice
        # are numpy ndarrays
        self._b_4D                      = False
        self._b_3D                      = False
        self._b_DICOM                   = False
        self._Vnp_4DVol                 = None
        self._Vnp_3DVol                 = None
        self._Mnp_2Dslice               = None
        self._dcm                       = None

        # A logger
        self._log                       = msg.Message()
        self._log.syslog(True)

        # Flags
        self._b_showSlices              = False
        self._b_convertMiddleSlice      = False
        self._b_convertMiddleFrame      = False
        self._b_reslice                 = False

        for key, value in kwargs.iteritems():
            if key == "inputFile":          self._str_inputFile         = value
            if key == "outputDir":          self._str_outputDir         = value
            if key == "outputFileStem":     self._str_outputFileStem    = value
            if key == "outputFileType":     self._str_outputFileType    = value
            if key == "sliceToConvert":     self._str_sliceToConvert    = value
            if key == "frameToConvert":     self._str_frameToConvert    = value
            if key == "showSlices":         self._b_showSlices          = value
            if key == 'reslice':            self._b_reslice             = value

        if self._str_frameToConvert.lower() == 'm':
            self._b_convertMiddleFrame = True
        elif self._str_frameToConvert:
            self._frameToConvert = int(self._str_frameToConvert)

        if self._str_sliceToConvert.lower() == 'm':
            self._b_convertMiddleSlice = True
        elif self._str_sliceToConvert:
            self._sliceToConvert = int(self._str_sliceToConvert)

        self._str_inputDir               = os.path.dirname(self._str_inputFile)
        if not self._str_inputDir: self._str_inputDir = '.'
        str_fileName, str_fileExtension  = os.path.splitext(self._str_outputFileStem)
        if self._str_outputFileType:
            str_fileExtension            = '.%s' % self._str_outputFileType

        if str_fileExtension and not self._str_outputFileType:
            self._str_outputFileType     = str_fileExtension

        if not self._str_outputFileType and not str_fileExtension:
            self._str_outputFileType     = '.png'

    def run(self):
        '''
        The main 'engine' of the class.
        '''
        raise NotImplementedError("Subclass responsibility")

    def dim_sliceSave(self, **kwargs):
        index   = 0
        frame   = 0
        subDir  = ""
        for key,val in kwargs.iteritems():
            if key == 'index':  index       = val 
            if key == 'frame':  frame       = val
            if key == 'subDir': str_subDir  = val
        
        if self._b_4D:
            str_outputFile = '%s/%s/%s-frame%03d-slice%03d.%s' % (
                                                    self._str_outputDir,
                                                    str_subDir,
                                                    self._str_outputFileStem,
                                                    frame, index,
                                                    self._str_outputFileType)
        else:
            str_outputFile = '%s/%s/%s-slice%03d.%s' % (
                                        self._str_outputDir,
                                        str_subDir,
                                        self._str_outputFileStem,
                                        index,
                                        self._str_outputFileType)
        self.slice_save(str_outputFile)

    def dim_save(self, **kwargs):
        dims            = self._Vnp_3DVol.shape
        self._log('Image volume logical (i, j, k) size: %s\n' % str(dims))
        str_dim         = 'z'
        b_makeSubDir    = False
        b_rot90         = False
        frame           = 0
        indexStart      = -1
        indexStop       = -1
        for key, val in kwargs.iteritems():
            if key == 'dimension':  str_dim         = val
            if key == 'makeSubDir': b_makeSubDir    = val
            if key == 'frame':      frame           = val
            if key == 'indexStart': indexStart      = val 
            if key == 'indexStop':  indexStop       = val
            if key == 'rot90':      b_rot90         = val
        
        str_subDir  = ''
        if b_makeSubDir: 
            str_subDir = str_dim
            misc.mkdir('%s/%s' % (self._str_outputDir, str_subDir))
        if str_dim == 'x':
            if indexStart == 0 and indexStop == -1:
                indexStop  = dims[0]
            for i in range(indexStart, indexStop):
                self._Mnp_2Dslice = self._Vnp_3DVol[i,:,:]
                if b_rot90: self._Mnp_2Dslice = np.rot90(self._Mnp_2Dslice)
                self.dim_sliceSave(index = i, subDir = str_subDir)
        if str_dim == 'y':
            if indexStart == 0 and indexStop == -1:
                indexStop  = dims[1]
            for j in range(indexStart, indexStop):
                self._Mnp_2Dslice = self._Vnp_3DVol[:,j,:]
                if b_rot90: self._Mnp_2Dslice = np.rot90(self._Mnp_2Dslice)
                self.dim_sliceSave(index = j, subDir = str_subDir)
        if str_dim == 'z':
            if indexStart == 0 and indexStop == -1:
                indexStop  = dims[2]
            for k in range(indexStart, indexStop):
                self._Mnp_2Dslice = self._Vnp_3DVol[:,:,k]
                if b_rot90: self._Mnp_2Dslice = np.rot90(self._Mnp_2Dslice)
                self.dim_sliceSave(index = k, subDir = str_subDir)

    def slice_save(self, astr_outputFile):
        '''
        Processes/saves a single slice.

        ARGS

        o astr_output
        The output filename to save the slice to.

        '''
        self._log('Outputfile = %s\n' % astr_outputFile)
        pylab.imsave(astr_outputFile, self._Mnp_2Dslice, cmap = cm.Greys_r)


class med2image_dcm(med2image):
    '''
    Sub class that handles DICOM data.
    '''
    def __init__(self, **kwargs):
        med2image.__init__(self, **kwargs)

        self.l_dcmFileNames = sorted(glob.glob('%s/*.dcm' % self._str_inputDir))
        self.slices         = self.l_dcmFileNames

        if self._b_convertMiddleSlice:
            self._sliceToConvert = int(self.slices/2)
            self._dcm            = dicom.read_file(self.l_dcmFileNames[self._sliceToConvert])
            self._str_inputFile  = self.l_dcmFileNames[self._sliceToConvert]
            str_outputFile       = self.l_dcmFileNames[self._sliceToConvert]
            if not self._str_outputFileStem.startswith('%'):
                self._str_outputFileStem, ext = os.path.splitext(self.l_dcmFileNames[self._sliceToConvert])
        if not self._b_convertMiddleSlice and self._sliceToConvert != -1:
            self._dcm = dicom.read_file(self.l_dcmFileNames[self._sliceToConvert])
        else:
            self._dcm = dicom.read_file(self._str_inputFile)
        if self._sliceToConvert == -1:
            self._b_3D = True
            self._dcm = dicom.read_file(self._str_inputFile)
            image = self._dcm.pixel_array
            shape2D = image.shape
            #print(shape2D)
            self._Vnp_3DVol = np.empty( (shape2D[0], shape2D[1], self.slices) )
            i = 0
            for img in self.l_dcmFileNames:
                self._dcm = dicom.read_file(img)
                image = self._dcm.pixel_array
                #print('%s: %s\n' % (img, image.shape))
                try:
                    self._Vnp_3DVol[:,:,i] = image
                except Exception, e:
                    error.fatal(self, 'dcmInsertionFail', '\nFor input DICOM file %s\n%s\n' % (img, str(e)))
                i += 1
        if self._str_outputFileStem.startswith('%'):
            str_spec = self._str_outputFileStem
            self._str_outputFileStem = ''
            for key in str_spec.split('%')[1:]:
                str_fileComponent = ''
                if key == 'inputFile':
                    str_fileName, str_ext = os.path.splitext(self._str_inputFile) 
                    str_fileComponent = str_fileName
                else:
                    str_fileComponent = eval('self._dcm.%s' % key)
                    str_fileComponent = med2image.urlify(str_fileComponent)
                if not self._str_outputFileStem:
                    self._str_outputFileStem = str_fileComponent
                else:
                    self._str_outputFileStem = self._str_outputFileStem + '-' + str_fileComponent
        image = self._dcm.pixel_array
        self._Mnp_2Dslice = image

    def run(self):
        '''
        Runs the DICOM conversion based on internal state.
        '''
        self._log('Converting DICOM image.\n')
        self._log('PatientName:                                %s\n' % self._dcm.PatientName)
        self._log('PatientAge:                                 %s\n' % self._dcm.PatientAge)
        self._log('PatientSex:                                 %s\n' % self._dcm.PatientSex)
        self._log('PatientID:                                  %s\n' % self._dcm.PatientID)
        self._log('SeriesDescription:                          %s\n' % self._dcm.SeriesDescription)
        self._log('ProtocolName:                               %s\n' % self._dcm.ProtocolName)
        if self._b_convertMiddleSlice:
            self._log('Converting middle slice in DICOM series:    %d\n' % self._sliceToConvert)

        l_rot90 = [ True, True, False ]
        misc.mkdir(self._str_outputDir)
        if not self._b_3D:
            str_outputFile = '%s/%s.%s' % (self._str_outputDir,
                                        self._str_outputFileStem,
                                        self._str_outputFileType)
            self.slice_save(str_outputFile)
        if self._b_3D:
            rotCount = 0
            if self._b_reslice:
                for dim in ['x', 'y', 'z']:
                    self.dim_save(dimension = dim, makeSubDir = True, rot90 = l_rot90[rotCount], indexStart = 0, indexStop = -1)
                    rotCount += 1
            else:
                self.dim_save(dimension = 'z', makeSubDir = False, rot90 = False, indexStart = 0, indexStop = -1)
                
class med2image_nii(med2image):
    '''
    Sub class that handles NIfTI data.
    '''

    def __init__(self, **kwargs):
        med2image.__init__(self, **kwargs)
        nimg = nib.load(self._str_inputFile)
        data = nimg.get_data()
        if data.ndim == 4:
            self._Vnp_4DVol     = data
            self._b_4D          = True
        if data.ndim == 3:
            self._Vnp_3DVol     = data
            self._b_3D          = True

    def run(self):
        '''
        Runs the NIfTI conversion based on internal state.
        '''

        self._log('About to perform NifTI to %s conversion...\n' %
                  self._str_outputFileType)

        frames     = 1
        frameStart = 0
        frameEnd   = 0

        sliceStart = 0
        sliceEnd   = 0

        if self._b_4D:
            self._log('4D volume detected.\n')
            frames = self._Vnp_4DVol.shape[3]
        if self._b_3D:
            self._log('3D volume detected.\n')

        if self._b_convertMiddleFrame:
            self._frameToConvert = int(frames/2)

        if self._frameToConvert == -1:
            frameEnd    = frames
        else:
            frameStart  = self._frameToConvert
            frameEnd    = self._frameToConvert + 1

        for f in range(frameStart, frameEnd):
            if self._b_4D:
                self._Vnp_3DVol = self._Vnp_4DVol[:,:,:,f]
            slices     = self._Vnp_3DVol.shape[2]
            if self._b_convertMiddleSlice:
                self._sliceToConvert = int(slices/2)

            if self._sliceToConvert == -1:
                sliceEnd    = -1
            else:
                sliceStart  = self._sliceToConvert
                sliceEnd    = self._sliceToConvert + 1

            misc.mkdir(self._str_outputDir)
            if self._b_reslice:
                for dim in ['x', 'y', 'z']:
                    self.dim_save(dimension = dim, makeSubDir = True, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True)
            else:
                self.dim_save(dimension = 'z', makeSubDir = False, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True)

