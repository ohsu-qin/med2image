# System imports
import os
import sys
import glob
import numpy as np
from random import randint
import re

# System dependency imports
import nibabel as nib
import dicom
import pylab
import matplotlib.cm as cm

# Project specific imports
from . import error
from . import message as msg
from ._common import systemMisc as misc
from ._common._colors import Colors


class ArgumentError(Exception):
    '''Input argument missing or invalid.'''
    pass


class DICOMError(Exception):
    '''DICOM file read error.'''
    pass


def run(inputLocation, **kwargs):
    '''
    Convert the image as specified by the given arguments.
    
    :param inputLocation: the input file or directory
    :param kwargs: the command line options
    :return: the converted output file location
    '''
    # Infer the output file stem and extension.
    outputFileStemOpt = kwargs.pop('outputFileStem', None)
    if outputFileStemOpt:
        stem, outputFileExtension = os.path.splitext(outputFileStemOpt)
        kwargs['outputFileStem'] = stem
        if outputFileExtension:
            # Elide the leading separator.
            outputFileExtension = outputFileExtension[1:]
    else:
        outputFileExtension = 'png'
    # Set the output file type and extension arguments.
    kwargs['outputFileExtension'] = outputFileExtension
    if 'outputFileType' not in kwargs:
        kwargs['outputFileType'] = outputFileExtension

    inputFileType = kwargs.pop('inputFileType', None)
    if os.path.isdir(inputLocation):
        # Only a DICOM directory is supported.
        if inputFileType:
            if inputFileType != 'dcm':
                raise ArgumentError("The input file type %s is not supported"
                                    "for a directory" % inputFileType)
        else:
            inputFileType = 'dcm'
    else:
        # The default output file name stem is taken from the
        # input file.
        if not outputFileStemOpt:
            _, filename = os.path.split(inputLocation)
            stem, ext = os.path.splitext(filename)
            if ext:
                # Remove a residual extension, if any, e.g. '.nii'
                # for file 'image.nii.gz'.
                stem, _ = os.path.splitext(stem)
            kwargs['outputFileStem'] = stem
        # The default input file type is inferred from the extension.
        if not inputFileType:
            stem, ext = os.path.splitext(inputLocation)
            if ext == '.gz':
                # The image extension of a compressed file precedes the
                # trailing '.gz' extension.
                _, ext = os.path.splitext(stem)
            if not ext:
                raise ArgumentError("The input file type could not be"
                                    " determined for file %s" % inputLocation)
            inputFileType = ext[1:]

    if inputFileType == 'nii':
        converter = med2image_nii(inputLocation, **kwargs)
    elif inputFileType == 'dcm':
        converter = med2image_dcm(inputLocation, **kwargs)
    else:
        raise NotImplementedError("The input file type is not supported: %s" %
                                  inputFileType)

    # And now run it!
    converter.run()


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

    @staticmethod
    def urlify(astr, ajoin = '_'):
        # Remove all non-word characters (everything except numbers and letters)
        astr = re.sub(r"[^\w\s]", '', astr)
        
        # Replace all runs of whitespace with an underscore
        astr = re.sub(r"\s+", ajoin, astr)
        
        return astr

    def __init__(self, **kwargs):
        # Directory and filenames
        self._outputFileStem = kwargs.get("outputFileStem")
        self._outputFileType = kwargs.get("outputFileType")
        self._outputDir = kwargs.get("outputDir", '.')

        self._reslice = kwargs.get("reslice")

        # A logger
        self._log = msg.Message()
        self._log.syslog(True)
        self._verbose = kwargs.get("verbose")

        self._convertMiddleFrame = False
        self._frameToConvert = None
        frameOpt = kwargs.get("frameToConvert")
        if frameOpt:
            if frameOpt.lower() == 'm':
                self._convertMiddleFrame = True
            else:
                self._frameToConvert = int(frameOpt)

        self._convertMiddleSlice = False
        self._sliceToConvert = None
        sliceOpt = kwargs.get("sliceToConvert")
        if sliceOpt:
            if sliceOpt.lower() == 'm':
                self._convertMiddleSlice = True
            else:
                self._sliceToConvert = int(sliceOpt)

        if self._outputFileType:
            fileExtension = '.%s' % self._outputFileType
        else:
            _, fileExtension = os.path.splitext(self._outputFileStem)
            if fileExtension:
                self._outputFileType = fileExtension[1:]
            else:
                self._outputFileType = 'png'

    def run(self):
        '''
        The main 'engine' of the class.
        '''
        raise NotImplementedError("Subclass responsibility")

    def dim_sliceSave(self, outputDir, index=0, frame=0):
        if self._is_4D:
            outputFile = ('%s/%s-frame%03d-slice%03d.%s' %
                          (outputDir, self._outputFileStem,
                           frame, index, self._outputFileType))
        else:
            outputFile = ('%s/%s-slice%03d.%s' %
                          (outputDir, self._outputFileStem,
                           index, self._outputFileType))
        self.slice_save(outputFile)

    def dim_save(self, **kwargs):
        dims = self._3DVol.shape
        if self._verbose:
            self._log('Image volume logical (i, j, k) size: %s\n' % str(dims))
        dim = kwargs.get('dimension', 'z')
        makeSubDir = kwargs.get('makeSubDir')
        frame = kwargs.get('frame', 0)
        indexStart = kwargs.get('indexStart', -1) 
        indexStop = kwargs.get('indexStop', -1)
        rot90 = kwargs.get('rot90')
        
        outputDir = self._outputDir
        if makeSubDir:
            outputDir += '/%s' % dim
        misc.mkdir(outputDir)
        if dim == 'x':
            if indexStart == 0 and indexStop == -1:
                indexStop = dims[0]
            for i in range(indexStart, indexStop):
                self._2Dslice = self._3DVol[i,:,:]
                if rot90: self._2Dslice = np.rot90(self._2Dslice)
                self.dim_sliceSave(outputDir, index=i)
        if dim == 'y':
            if indexStart == 0 and indexStop == -1:
                indexStop = dims[1]
            for j in range(indexStart, indexStop):
                self._2Dslice = self._3DVol[:,j,:]
                if rot90: self._2Dslice = np.rot90(self._2Dslice)
                self.dim_sliceSave(outputDir, index=k)
        if dim == 'z':
            if indexStart == 0 and indexStop == -1:
                indexStop = dims[2]
            for k in range(indexStart, indexStop):
                self._2Dslice = self._3DVol[:,:,k]
                if rot90: self._2Dslice = np.rot90(self._2Dslice)
                self.dim_sliceSave(outputDir, index=k)

    def slice_save(self, outputFile):
        '''
        Processes/saves a single slice.

        ARGS

        o outputFile
        The output filename to save the slice to.

        '''
        if self._verbose:
            self._log('Output file: %s\n' % outputFile)
        pylab.imsave(outputFile, self._2Dslice, cmap = cm.Greys_r)


class med2image_dcm(med2image):
    '''
    Subclass that handles DICOM data.
    '''

    def __init__(self, inputLocation, **kwargs):
        super(med2image_dcm, self).__init__(**kwargs)

        self._is_3D = self._is_4D = False
        if os.path.isfile(inputLocation):
            dcm = dicom.read_file(inputLocation)
            self._2Dslice = dcm.pixel_array
        else:
            fileNames = sorted(glob.glob('%s/*' % inputLocation))
            slice_cnt = len(fileNames)
            if self._convertMiddleSlice:
                self._sliceToConvert = int(slice_cnt/2)
            if self._sliceToConvert == None:
                # Build a 3D volume from the input DICOM files.
                self._is_3D = True
                for i, fileName in enumerate(fileNames):
                    dcm = dicom.read_file(fileName)
                    image = dcm.pixel_array
                    if i == 0:
                        if self._outputFileStem:
                            self._format_output_stem(dcm)
                        else:
                             self._outputFileStem = 'image'
                        shape2D = dcm.pixel_array.shape
                        shape3D = (shape2D[0], shape2D[1], slice_cnt)
                        self._3DVol = np.empty(shape3D)
                    # If there is an image inserting the 2D image,
                    # let the error and trace stack propagate up
                    # the call chain.
                    self._3DVol[:,:,i] = image
            else:
                inputFile = self.fileNames[self._sliceToConvert]
                dcm = dicom.read_file(inputFile)
                self._2Dslice = dcm.pixel_array
                if self._outputFileStem:
                    self._format_output_stem(dcm)
                else:
                    _, stem = os.path.split(inputFile)
                    stem, ext = os.path.splitext(stem)
                    if ext == '.gz':
                        stem, _ = os.path.splitext(stem)
                    self._outputFileStem = stem
    
    def _format_output_stem(self, dcm):
        if not self._outputFileStem:
            return
        # Maintain backward compatibility for undelimited meta attributes.
        spec = self._outputFileStem.split('%')
        delimit = lambda s: s if not s or s[0] == '(' else '(%s)' % s
        delimited = '%'.join((delimit(s) for s in spec))
        # The meta substitutions.
        attrs = re.findall(r'%\((\w+)\)', delimited)
        sub = lambda s, attr: s.replace('%%(%s)' % attr,
                                        str(getattr(dcm, attr)))
        self._outputFileStem = reduce(sub, attrs, delimited)

    def run(self):
        '''
        Runs the DICOM conversion based on internal state.
        '''
        if self._verbose:
            self._log('Converting DICOM image.\n')
        if self._convertMiddleSlice:
            if self._verbose:
                self._log('Converting middle slice in DICOM series:    %d\n' % self._sliceToConvert)

        l_rot90 = [ True, True, False ]
        misc.mkdir(self._outputDir)
        if self._is_3D:
            rotCount = 0
            if self._reslice:
                for dim in ['x', 'y', 'z']:
                    self.dim_save(dimension = dim, makeSubDir = True, rot90 = l_rot90[rotCount], indexStart = 0, indexStop = -1)
                    rotCount += 1
            else:
                self.dim_save(dimension = 'z', makeSubDir = False, rot90 = False, indexStart = 0, indexStop = -1)
        else:
            outputFile = '%s/%s.%s' % (self._outputDir,
                                        self._outputFileStem,
                                        self._outputFileType)
            self.slice_save(outputFile)
                
class med2image_nii(med2image):
    '''
    Subclass that handles NIfTI data.
    '''

    def __init__(self, inputLocation, **kwargs):
        super(med2image_dcm, self).__init__(**kwargs)
        nimg = nib.load(inputLocation)
        # The actual data volume and slice
        # are numpy ndarrays
        data = nimg.get_data()
        if data.ndim == 4:
            self._4DVol = data
            self._is_3D = False
            self._is_4D = True
        elif data.ndim == 3:
            self._3DVol = data
            self._is_3D = True
            self._is_4D = False

    def run(self):
        '''
        Runs the NIfTI conversion based on internal state.
        '''

        if self._verbose:
            self._log('About to perform NifTI to %s conversion...\n' %
                      self._outputFileType)

        if self._is_4D:
            if self._verbose:
                self._log('4D volume detected.\n')
            frames = self._4DVol.shape[3]
        else:
            frames = 1
        if self._verbose and self._is_3D:
            self._log('3D volume detected.\n')

        if self._convertMiddleFrame:
            self._frameToConvert = int(frames/2)

        if self._frameToConvert == None:
            frameStart = 0
            frameEnd = frames
        else:
            frameStart = self._frameToConvert
            frameEnd = self._frameToConvert + 1

        for f in range(frameStart, frameEnd):
            if self._is_4D:
                self._3DVol = self._4DVol[:,:,:,f]
            slices = self._3DVol.shape[2]
            if self._convertMiddleSlice:
                self._sliceToConvert = int(slices/2)

            if self._sliceToConvert == None:
                sliceStart = 0
                sliceEnd = -1
            else:
                sliceStart = self._sliceToConvert
                sliceEnd = self._sliceToConvert + 1

            misc.mkdir(self._outputDir)
            if self._reslice:
                for dim in ['x', 'y', 'z']:
                    self.dim_save(dimension = dim, makeSubDir = True, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True)
            else:
                self.dim_save(dimension = 'z', makeSubDir = False, indexStart = sliceStart, indexStop = sliceEnd, rot90 = True)


