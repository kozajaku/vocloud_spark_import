import logging
import warnings
import astropy.convolution as convolution
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
import StringIO
import vocloud_spark_preprocess.util as utils
import pandas as pd
import astropy.io.fits as pyfits
from pyspark.mllib.feature import PCA
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
import sklearn.preprocessing as prep


__author__ = 'Andrej Palicka <andrej.palicka@merck.com>'


logger = logging.getLogger(__name__)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def parse_votable(file_path, content):
    """
    :param file: The file to parse
    :return: Returns a pandas Series, where the index are the waves and values are intensities
    """
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

def resample(spectrum, resampled_header_broadcast, label_col=None, convolve=False, normalize=True):
    """Resamples the spectrum so that the x-axis starts at low and ends at high, while
    keeping the delta between the wavelengths"""
    resampled_header = resampled_header_broadcast.value
    if label_col is not None:
        logger.debug(spectrum.columns)
        without_label = spectrum.drop(label_col, axis=1)
        without_label.columns = pd.to_numeric(without_label.columns, errors="ignore")
    else:
        without_label = spectrum
    if convolve:
        to_interpolate = convolution.convolve(without_label.iloc[0].values, convolution.Gaussian1DKernel(7),
                                              boundary="extend")
    else:
        to_interpolate = without_label.iloc[0].values
    logger.debug(without_label)
    interpolated = np.interp(resampled_header, without_label.columns.values, to_interpolate)
    interpolated = interpolated[3:-3]  # remove some weird artefacts that might happen because of convo/interpolation
    if normalize:
            interpolated = prep.minmax_scale([interpolated], axis=1)
    logger.debug("Interpolated:%s", interpolated)
    interpolated_df = pd.DataFrame(data=interpolated, columns=resampled_header[3:-3], index=spectrum.index.values)
    if label_col is not None:
        interpolated_df[label_col] = spectrum[label_col]
    return interpolated_df


def parse_fits(path, content, low, high):
    str_file = StringIO.StringIO(content)
    with pyfits.open(str_file) as hdu_list:
        hdu = hdu_list[0]
        if hdu.header["CLASS"].lower() != "star":
            return None
        crval1, crpix1 = hdu.header["CRVAL1"], hdu.header["CRPIX1"]
        cdelt1 = hdu.header["CD1_1"]
        name = os.path.basename(os.path.splitext(path)[0])
        def specTrans(pixNo):
            return 10**(crval1+(pixNo+1-crpix1)*cdelt1)
        return pd.DataFrame.from_records({specTrans(spec): flux for spec, flux in enumerate(hdu.data[2]) if low < specTrans(spec) < high}, index=[name])

def parse_spectra_file(file_path, content, low, high):
    ext = os.path.splitext(file_path)[1]
    if ext == ".vot":
        return parse_votable(file_path, content)
    elif ext == ".fits":
        return parse_fits(file_path, content, low, high)
    else:
        raise ValueError("Only votable and fits files are supported for now")


def high_low_op(acc, x):
    logger.info(x)
    w_low = max(acc[0], x.columns[0])
    w_high = min(acc[1], x.columns[-1])
    return w_low, w_high

def high_low_comb(acc1, acc2):
    w_low = max(acc1[0], acc2[0])
    w_high = min(acc1[1], acc2[1])
    return w_low, w_high


def transform_pca(x):
    columns = x.columns
    if x.columns[-1] == 'label':
        columns = columns[:-1]
    return x[columns].iloc[0].values


def preprocess(sc, files_rdd, labeled_spectra, cut, label=True, **kwargs):
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
    # TODO support archives
    cut_low = cut['low']
    cut_high = cut['high']
    spectra = files_rdd.map(lambda x: parse_spectra_file(x[0], x[1], cut_low, cut_high)).filter(lambda x: x is not None).cache()
    low, high = spectra.union(labeled_spectra.map(lambda x: x.drop(x.columns[-1], axis=1))).aggregate((0.0, sys.float_info.max), high_low_op, high_low_comb)
    mean_step = spectra.map(lambda x: x.columns[1] - x.columns[0]).mean()
    logger.debug("low %f high %f %f", low, high, mean_step)
    resampled_header = sc.broadcast(np.arange(low, high, mean_step))
    spectra = spectra.map(lambda x: resample(x, resampled_header_broadcast=resampled_header, normalize=kwargs.get('minmax_scale')))
    if label:
        spectra = spectra.map(lambda x: x.assign(label=pd.Series([-1], index=x.index)))

    if labeled_spectra is not None:
        spectra = labeled_spectra.map(lambda x: resample(x, resampled_header_broadcast=resampled_header, label_col="label" if label else None,
                                                         convolve=True, normalize=kwargs.get('minmax_scale'))).union(spectra).repartition(kwargs.get("partitions", 100))

    if kwargs.get('pca') is not None:
        namesByRow = spectra.zipWithIndex().map(lambda s: (s[1], (s[0].index, s[0]['label'].iloc[0])) if label else (s[1], s[0].index))
        logger.info("Doing PCA")
        pca_params = kwargs['pca']
        k = pca_params.get("k", 10)
        pca = PCA(k)
        fitted_pca = pca.fit(spectra.map(lambda x: Vectors.dense(x[x.columns[:-1]].iloc[0].values)))
        transformed_spectra = fitted_pca.transform(spectra.
                                       map(lambda x: transform_pca(x))).zipWithIndex(). \
            map(lambda x: (x[1], x[0])).join(namesByRow)
        spectra = transformed_spectra.map(lambda x: pd.DataFrame(data=[x[1][0].tolist() +
                                       ([x[1][1][1]] if label else [])],
                                       index=x[1][1][0] if label else x[2],
                                       columns=range(k) + ['label'] if label else range(k)))

    return resampled_header.value, spectra.sortBy(lambda x: x['label'].values[0], ascending=False,
                                                  numPartitions=kwargs.get("partitions", 100))
