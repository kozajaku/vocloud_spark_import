#!/usr/bin/env python3
import logging
import warnings
from functools import partial, partialmethod
import numpy as np
import astropy.io.votable as vo
import astropy.convolution as convolution
import os
import pandas as pd
import sys
import pyspark

__author__ = 'Andrej Palicka <andrej.palicka@merck.com>'

class Preprocessing(object):

    def __init__(self):
        """

        :param sc: The SparkContext object
        """
        self.logger = logging.getLogger(__name__)

    def parse_votable(self, file):
        """
        :param file: The file to parse
        :return: Returns a pandas Series, where the index are the waves and values are intensities
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = vo.parse(file)
        spectrum = parsed.get_first_table().to_table().to_pandas()
        series = pd.DataFrame(data=[spectrum["flux"].tolist()], columns=spectrum["spectral"].tolist(),
                              index=[os.path.splitext(os.path.basename(file))[0]])
        return series

    def resample(self, spectrum: pd.DataFrame, low: float, high: float, step: float, label_col=None):
        """Resamples the spectrum so that the x-axis starts at low and ends at high, while
        keeping the delta between the wavelengths"""
        resampled_header = np.arange(low, high, step)
        if label_col is not None:
            self.logger.debug(spectrum.columns)
            without_label = spectrum.drop(label_col, axis=1)
            without_label.columns = pd.to_numeric(without_label.columns, errors="ignore")
        else:
            without_label = spectrum
        convolved = convolution.convolve(without_label.iloc[0].values, convolution.Gaussian1DKernel(1))
        self.logger.debug(convolved)
        interpolated = np.interp(resampled_header, without_label.columns.values, convolved)
        self.logger.debug("Interpolated:%s", interpolated)
        interpolated_df = pd.DataFrame(data=[interpolated], columns=resampled_header, index=[spectrum.index.values])
        if label_col is not None:
            interpolated_df[label_col] = spectrum[label_col]
        return interpolated_df

    def preprocess(self, files_rdd: pyspark.RDD, labeled_spectra: pyspark.RDD, label=True):
        """
        :param path: A path to the input data. It should be a directory containing the votable or FITS files.
        :param labeled_path: A path to the CSV file with spectra already labeled. These shall be resampled so that
         the have the same resolution as the unlabeled spectra. They shall undergo the same preprocessing as the rest.
        :param label: Set to False if you want to omit the label from the output.
        :return: A RDD DataFrame with the preprocessed spectra. It shall contain the labeled spectra at the beginning,
        followed by the unlabeled, whose label shall be set to -1. In case you *label* was set to False, the output
        shall not contain any label and the ordering shall be arbitrary.
        """
        self.logger.info("Starting preprocessing")

        def parse_spectra_file(file_path):
            if os.path.splitext(file_path)[1] == ".vot":
                return self.parse_votable(file_path)
            else:
                raise ValueError("Only votable files are supported for now")

        def high_low_op(acc, x: pd.DataFrame):
            w_low = max(acc[0], x.columns[0])
            w_high = min(acc[1], x.columns[-1])
            return w_low, w_high

        def high_low_comb(acc1, acc2):
            w_low = max(acc1[0], acc2[0])
            w_high = min(acc1[1], acc2[1])
            return w_low, w_high



        # TODO support archives
        spectra = files_rdd.map(lambda x: parse_spectra_file(x[0])).cache()
        low, high = spectra.aggregate((0.0, sys.float_info.max), high_low_op, high_low_comb)
        mean_step = spectra.map(lambda x: x.columns[1] - x.columns[0]).mean()
        self.logger.debug("low %f high %f %f", low, high, mean_step)
        spectra = spectra.map(lambda x: self.resample(x, low=low, high=high, step=mean_step)).cache()
        if label:
            spectra = spectra.map(lambda x: x.assign(label=pd.Series([-1], index=x.index)))
        if labeled_spectra is not None:
            spectra = labeled_spectra.map(lambda x: self.resample(x, low=low, high=high, step=mean_step,
                                                                  label_col="label" if label else None)).union(spectra)
        return spectra
