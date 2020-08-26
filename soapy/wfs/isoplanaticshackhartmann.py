
from . import shackhartmann
from .. import numbalib, AOFFT, lineofsight

from astropy.io import fits
import numpy

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32


class ExtendedSH(shackhartmann.ShackHartmann):
    """
    Extended source Shack-Hartmann WFS.
    """
    def calcInitParams(self):
        super().calcInitParams()

        # Add in the extended object
        self.reference_image = self.wfsConfig.referenceImage

        if type(self.wfsConfig.extendedObject) is str:
            self.extended_object = fits.getdata(self.wfsConfig.extendedObject)
        else:
            self.extended_object = self.wfsConfig.extendedObject

    def allocDataArrays(self):
        super().allocDataArrays()

        self.psfs = self.subap_focus_intensity.copy()

        self.ExtendedConvolver = AOFFT.Convolve(self.psfs.shape, self.extended_object.shape,
                                                threads=self.wfsConfig.fftwThreads, axes=(-2, -1))

        self.corrSubapArrays = numpy.zeros(self.centSubapArrays.shape, dtype=DTYPE)

    def initLos(self):
        self.los = lineofsight.ExtendedLineOfSight(self.config, self.soapy_config, propagationDirection="down")

    def getStatic(self):
        # TODO: Implement this from Shack-Hartmann
        pass

    def calcFocalPlane(self, intensity=1):
        '''
        Calculates the wfs focal plane, given the phase across the WFS

        Parameters:
            intensity (float): The relative intensity of this frame, is used when multiple WFS frames taken for extended sources.
        '''

        if self.config.propagationMode=="Geometric":
            # Have to make phase the correct size if geometric prop
            numbalib.wfslib.zoomtoefield(self.los.phase, self.interp_efield)

        else:
            self.interp_efield = self.EField

        # Create an array of individual subap EFields
        self.fft_input_data[:] = 0
        numbalib.wfslib.chop_subaps_mask(self.interp_efield, self.interp_subap_coords, self.nx_subap_interp,
                                         self.fft_input_data, self.scaledMask)
        self.fft_input_data[:, :self.nx_subap_interp, :self.nx_subap_interp] *= self.tilt_fix_efield
        self.FFT()

        self.temp_subap_focus = AOFFT.ftShift2d(self.fft_output_data)

        numbalib.abs_squared(self.temp_subap_focus, self.psfs)

        if intensity != 1:
            self.psfs *= intensity

        self.subap_focus_intensity[:] = self.ExtendedConvolver(self.psfs, self.extended_object).real

    def integrateDetectorPlane(self):
        '''
                Scales and bins intensity data onto the detector with a given number of
                pixels.

                If required, will first convolve final PSF with LGS PSF, then bin
                PSF down to detector size. Finally puts back into ``wfsFocalPlane``
                array in correct order.
                '''

        # Apply focal plane mask to stop sub-apertures overlapping
        if self.SUBAP_OVERSIZE != 1:
            print("Applying mask")
            coord = int(self.subapFFTPadding / (2 * self.SUBAP_OVERSIZE))
            self.fieldMask = numpy.zeros((self.subapFFTPadding,) * 2)
            self.fieldMask[coord:-coord, coord:-coord] = 1

            self.subap_focus_intensity *= self.fieldMask

        # bins back down to correct size and then fits them back in to a focal plane array
        self.binnedFPSubapArrays[:] = 0
        numbalib.wfslib.bin_imgs(
            self.subap_focus_intensity, self.config.fftOversamp, self.binnedFPSubapArrays,
        )

        # Scale each sub-ap flux by sub-aperture fill-factor
        self.binnedFPSubapArrays \
            = (self.binnedFPSubapArrays.T * self.subapFillFactor).T

        numbalib.wfslib.place_subaps_on_detector(
            self.binnedFPSubapArrays, self.detector, self.detector_subap_coords, self.valid_subap_coords)

    def zeroPhaseData(self):
        # TODO: Implement this in loop for Lines of sight
        pass

    def frame(self, scrns, phase_correction=None, read=True, iMatFrame=False):
        # TODO: Implement the framing using multiple FOVs
        pass

    def calculateSlopes(self):
        # TODO :Implement this for extended FOV WFS
        pass

    @property
    def EField(self):
        # TODO: Find the central EField to return? Or return them all
        return self.los.EField

    @EField.setter
    def EField(self, EField):
        # TODO: How do we get around this...
        self.los.EField = EField