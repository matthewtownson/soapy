"""
A Shack Hartmann WFS for use with extended reference sources, such as solar AO, where correlation centroiding techniques are required.

"""

import numpy
import scipy.signal
import pyfftw

try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("PyAOS requires either pyfits or astropy")

from aotools.image_processing import centroiders

from .. import AOFFT, logger
from . import shackhartmann
from .. import numbalib
from .. import lineofsight

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

class ExtendedSH(shackhartmann.ShackHartmann):

    def calcInitParams(self):
        super(ExtendedSH, self).calcInitParams()

        # For correlation centroider, open reference image.
        self.referenceImage = self.wfsConfig.referenceImage

    def initFFTs(self):
        """
        Initialise an extra FFT for the convolution in the correlation
        """
        super(ExtendedSH, self).initFFTs()

        # self.corrFFT = AOFFT.FFT(
        #         inputSize=(
        #             self.activeSubaps, self.wfsConfig.pxlsPerSubap,
        #             self.wfsConfig.pxlsPerSubap),
        #         axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
        #         THREADS=self.wfsConfig.fftwThreads,
        #         fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"),
        #         )

        # Init the FFT to the focal plane
        # self.FFT = AOFFT.FFT(
        #         inputSize=(
        #             self.n_subaps, self.subapFFTPadding, self.subapFFTPadding),
        #         axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
        #         THREADS=self.threads,
        #         fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT"))

        self.corr_fft_input_data = pyfftw.empty_aligned(
            (self.n_subaps, self.wfsConfig.pxlsPerSubap, self.wfsConfig.pxlsPerSubap), dtype=CDTYPE)
        self.corr_fft_output_data = pyfftw.empty_aligned(
            (self.n_subaps, self.wfsConfig.pxlsPerSubap, self.wfsConfig.pxlsPerSubap), dtype=CDTYPE)
        self.corrFFT = pyfftw.FFTW(
            self.corr_fft_input_data, self.corr_fft_output_data, axes=(-2, -1),
            threads=self.threads, flags=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT")
        )

        self.corr_ifft_input_data = pyfftw.empty_aligned(
            (self.n_subaps, self.wfsConfig.pxlsPerSubap, self.wfsConfig.pxlsPerSubap), dtype=CDTYPE)
        self.corr_ifft_output_data = pyfftw.empty_aligned(
            (self.n_subaps, self.wfsConfig.pxlsPerSubap, self.wfsConfig.pxlsPerSubap), dtype=CDTYPE)
        self.corrIFFT = pyfftw.FFTW(
            self.corr_ifft_input_data, self.corr_ifft_output_data, axes=(-2, -1),
            threads=self.threads, flags=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"), direction="FFTW_BACKWARD"
        )

        # self.corrIFFT = AOFFT.FFT(
        #         inputSize=(
        #             self.activeSubaps, self.wfsConfig.pxlsPerSubap,
        #             self.wfsConfig.pxlsPerSubap),
        #         axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
        #         THREADS=self.wfsConfig.fftwThreads,
        #         fftw_FLAGS=(self.wfsConfig.fftwFlag,"FFTW_DESTROY_INPUT"),
        #         direction="BACKWARD")

        # Also open object if its given
        self.extendedObject = self.wfsConfig.extendedObject

    def allocDataArrays(self):
        super(ExtendedSH, self).allocDataArrays()

        # Make a convolution object to apply the object
        if self.extendedObject is None:
            self.objectConv = None
        else:
            self.objectConv = AOFFT.Convolve(
                    self.binnedFPSubapArrays.shape, self.extendedObject.shape,
                    threads=self.wfsConfig.fftwThreads, axes=(-2, -1)
                    )

        self.corrSubapArrays = numpy.zeros(self.centSubapArrays.shape, dtype=DTYPE)

    def integrateDetectorPlane(self):
        """
        If an extended object is supplied, convolve with spots to make
        the detector images
        """
        if self.extendedObject is not None:
            # Perform correlation to get subap images
            self.binnedFPSubapArrays[:] = self.objectConv(
                    self.binnedFPSubapArrays, self.extendedObject).real

        # If sub-ap is oversized, apply field mask (TODO:make more general)
        if self.SUBAP_OVERSIZE != 1:
            coord = int(self.subapFFTPadding/(2*self.SUBAP_OVERSIZE))
            print(coord)
            fieldMask = numpy.zeros((self.subapFFTPadding,)*2)
            fieldMask[coord:-coord, coord:-coord] = 1

            self.binnedFPSubapArrays *= fieldMask

        # Finally, run put these arrays onto the simulated detector
        numbalib.wfslib.bin_imgs(
            self.subap_focus_intensity, self.config.fftOversamp, self.binnedFPSubapArrays,
        )

        # Scale each sub-ap flux by sub-aperture fill-factor
        self.binnedFPSubapArrays = (self.binnedFPSubapArrays.T * self.subapFillFactor).T

        numbalib.wfslib.place_subaps_on_detector(
            self.binnedFPSubapArrays, self.detector, self.detector_subap_coords, self.valid_subap_coords)

    def makeCorrelationImgs(self):
        """
        Use a convolution method to retrieve the 2d correlation peak between the subaperture and reference images.
        """

        # Remove the min from each sub-ap to increase contrast
        self.centSubapArrays[:] = (
                self.centSubapArrays.T-self.centSubapArrays.min((1,2))).T

        # Now do convolution
        # Get inverse FT of subaps
        # iCentSubapArrays = self.corrFFT(self.centSubapArrays)
        #
        # # Multiply by inverse of reference image FFT (made when set in property)
        # # Do FFT to get correlation
        # self.corrSubapArrays = self.corrIFFT(
        #         iCentSubapArrays*self.iReferenceImage).real

        # for i, subap in enumerate(self.centSubapArrays):
        #     self.corrSubapArrays[i] = scipy.signal.fftconvolve(subap, self.referenceImage[i], mode='same')

        if self.config.correlationFFTPad is None:
            subap_pad = self.centSubapArrays
            ref_pad = self.referenceImage
        else:
            PAD = round(0.5*(self.config.correlationFFTPad - self.config.pxlsPerSubap))
            subap_pad = numpy.pad(
                    self.centSubapArrays, mode='constant',
                    pad_width=((0,0), (PAD, PAD), (PAD, PAD)))
            ref_pad = numpy.pad(
                    self.referenceImage, mode='constant',
                    pad_width=((0,0), (PAD, PAD), (PAD, PAD)))

        self.corrSubapArrays = numpy.fft.fftshift(numpy.fft.ifft2(
                numpy.fft.fft2(subap_pad, axes=(1,2)) * numpy.fft.fft2(ref_pad, axes=(1,2)))).real


    @property
    def referenceImage(self):
        """
        A reference image to be used by a correlation centroider.
        """
        return self._referenceImage

    @referenceImage.setter
    def referenceImage(self, referenceImage):
        if referenceImage is not None:
            # If given value is a string, assume a filename of fits file
            if isinstance(referenceImage, str):
                referenceImage = fits.getdata(referenceImage)

            # Shape of expected ref values
            refShape = (
                    self.n_subaps, self.wfsConfig.pxlsPerSubap,
                    self.wfsConfig.pxlsPerSubap)
            self._referenceImage = numpy.zeros(refShape)

            # if its an array of sub-aps, no work needed
            if referenceImage.shape == refShape:
                self._referenceImage = referenceImage

            # If its the size of a sub-ap, set all subaps to that value
            elif referenceImage.shape == (self.wfsConfig.pxlsPerSubap,)*2:
                # Make a placeholder for the reference image
                self._referenceImage = numpy.zeros(
                        (self.n_subaps, self.wfsConfig.pxlsPerSubap,
                        self.wfsConfig.pxlsPerSubap))
                self._referenceImage[:] = referenceImage

            # If its the size of the detector, assume its a tiled array of sub-aps
            elif referenceImage.shape == (self.detectorPxls,)*2:

                for i, (x, y) in enumerate(self.detectorSubapCoords):
                    self._referenceImage[i] = referenceImage[
                            x:x+self.wfsConfig.pxlsPerSubap,
                            y:y+self.wfsConfig.pxlsPerSubap]

            # Do the FFT of the reference image for the correlation
            self.iReferenceImage = numpy.fft.ifft2(
                    self._referenceImage, axes=(1,2))
        else:
            self._referenceImage = None

    def initLos(self):
        self.los = lineofsight.ExtendedLineOfSight(self.config, self.soapy_config, propagationDirection="down")

    @property
    def extendedObject(self):
        return self._extendedObject

    @extendedObject.setter
    def extendedObject(self, extendedObject):
        if extendedObject is not None:
            # If a string, assume a fits file
            if isinstance(extendedObject, str):
                extendedObject = fits.getdata(extendedObject)

            if extendedObject.shape!=(self.subapFFTPadding,)*2:
                raise ValueError("Shape of extended object must be ({}, {}). This is `pxlsPersubap * fftOversamp`".format(self.subapFFTPadding, self.subapFFTPadding))

            self._extendedObject = extendedObject
        else:
            self._extendedObject = None
