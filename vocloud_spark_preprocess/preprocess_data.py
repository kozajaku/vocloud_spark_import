import logging
import warnings
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
import StringIO
import vocloud_spark_preprocess.util as utils


__author__ = 'Andrej Palicka <andrej.palicka@merck.com>'


logger = logging.getLogger("py4j")

def parse_votable(file_path, content):
    """
    :param file: The file to parse
    :return: Returns a pandas Series, where the index are the waves and values are intensities
    """
    try:
        import pandas as pd
    except:
        utils.add_dependencies()
        import pandas as pd
    name = os.path.basename(os.path.splitext(file_path)[0])
    tree = ET.fromstring(content)
    spectral_col = []
    flux_col = []
    for child in tree.iter(tag="{http://www.ivoa.net/xml/VOTable/v1.2}TR"):
        values = child.findall("{http://www.ivoa.net/xml/VOTable/v1.2}TD")
        spectral = float(values[0].text)
        flux = float(values[1].text)
        spectral_col.append(spectral)
        flux_col.append(flux)
    assert(len(flux_col) > 0)
    series = pd.DataFrame(data=[flux_col], columns=spectral_col,
                          index=[name])
    logger.debug(series)
    return series

def resample(spectrum, low, high, step, label_col=None, convolve=False):
    """Resamples the spectrum so that the x-axis starts at low and ends at high, while
    keeping the delta between the wavelengths"""
    try:
        import astropy.convolution as convolution
        import pandas as pd
    except ImportError:
        utils.add_dependencies()
        import astropy.convolution as convolution
        import pandas as pd
    resampled_header = np.arange(low, high, step)
    if label_col is not None:
        logger.debug(spectrum.columns)
        without_label = spectrum.drop(label_col, axis=1)
        without_label.columns = pd.to_numeric(without_label.columns, errors="ignore")
    else:
        without_label = spectrum
    if convolve:
        to_interpolate = convolution.convolve(without_label.iloc[0].values, convolution.Gaussian1DKernel(7))
    else:
        to_interpolate = without_label.iloc[0].values
    logger.debug(without_label)
    interpolated = np.interp(resampled_header, without_label.columns.values, to_interpolate, left=0.0, right=0.0)
    logger.debug("Interpolated:%s", interpolated)
    interpolated_df = pd.DataFrame(data=[interpolated], columns=resampled_header, index=[spectrum.index.values])
    if label_col is not None:
        interpolated_df[label_col] = spectrum[label_col]
    return interpolated_df


def parse_fits(path, content):
    try:
        import astropy.io.fits as pyfits
        import pandas as pd
    except ImportError:
        utils.add_dependencies()
        import astropy.io.fits as pyfits
        import pandas as pd
    str_file = StringIO.StringIO(content)
    with pyfits.open(str_file) as hdu_list:
        hdu = hdu_list[0]
        crval1, crpix1 = hdu.header["CRVAL1"], hdu.header["CRPIX1"]
        cdelt1 = hdu.header["CD1_1"]
        name = os.path.basename(os.path.splitext(path)[0])
        def specTrans(pixNo):
            return 10**(crval1+(pixNo+1-crpix1)*cdelt1)

        return pd.DataFrame.from_records({specTrans(spec): flux for spec, flux in enumerate(hdu.data[2])}, index=[name])

def parse_spectra_file(file_path, content):
    ext = os.path.splitext(file_path)[1]
    if ext == ".vot":
        return parse_votable(file_path, content)
    elif ext == ".fits":
        return parse_fits(file_path, content)
    else:
        raise ValueError("Only votable and fits files are supported for now")


def high_low_op(acc, x):
    logger.debug(x)
    w_low = max(acc[0], x.columns[0])
    w_high = min(acc[1], x.columns[-1])
    return w_low, w_high

def high_low_comb(acc1, acc2):
    w_low = max(acc1[0], acc2[0])
    w_high = min(acc1[1], acc2[1])
    return w_low, w_high


def preprocess(files_rdd, labeled_spectra, label=True):
    """
    :param path: A path to the input data. It should be a directory containing the votable or FITS files.
    :param labeled_path: A path to the CSV file with spectra already labeled. These shall be resampled so that
     the have the same resolution as the unlabeled spectra. They shall undergo the same preprocessing as the rest.
    :param label: Set to False if you want to omit the label from the output.
    :return: A RDD DataFrame with the preprocessed spectra. It shall contain the labeled spectra at the beginning,
    followed by the unlabeled, whose label shall be set to -1. In case you *label* was set to False, the output
    shall not contain any label and the ordering shall be arbitrary.
    """
    logger.info("Starting preprocessing")
    try:
        import pandas as pd
    except ImportError:
        utils.add_dependencies()
        import pandas as pd
    # TODO support archives
    spectra = files_rdd.map(lambda x: parse_spectra_file(x[0], x[1])).cache()
    low, high = spectra.union(labeled_spectra.map(lambda x: x.drop(x.columns[-1], axis=1))).aggregate((0.0, sys.float_info.max), high_low_op, high_low_comb)
    mean_step = spectra.map(lambda x: x.columns[1] - x.columns[0]).mean()
    logger.debug("low %f high %f %f", low, high, mean_step)
    spectra = spectra.map(lambda x: resample(x, low=low, high=high, step=mean_step)).cache()
    if label:
        spectra = spectra.map(lambda x: x.assign(label=pd.Series([-1], index=x.index)))
    if labeled_spectra is not None:
        spectra = labeled_spectra.map(lambda x: resample(x, low=low, high=high, step=mean_step,
                                                              label_col="label" if label else None,
                                                              convolve=True)).union(spectra).\
            sortBy(lambda x: x['label'].values[0], ascending=False)
    return spectra
