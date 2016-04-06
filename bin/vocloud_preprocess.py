import sys
# fix for spark py-files zip containig dynamic libraries like .so
dep_zip_list = [i for i, val in enumerate(sys.path) if val.endswith("dependencies.zip")]
if len(dep_zip_list) > 0:
    import zipfile
    import tempfile
    dep_zip = sys.path[dep_zip_list[0]]
    out_dir = tempfile.mkdtemp()
    with open(dep_zip, 'rb') as dep_zip_file:
        z = zipfile.ZipFile(dep_zip_file)
        for name in z.namelist():
            z.extract(name, out_dir)
    sys.path.insert(0, out_dir)
import os
import argparse
import pandas as pd
#import matplotlib.pyplot as plt
import vocloud_spark_preprocess.preprocess_data as prep
import logging
import logging.config
from astropy.io.votable import parse
from pyspark import SparkConf, SparkContext
import json
import requests
try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode


__author__ = 'Andrej Palicka <andrej.palicka@merck.com>'

BASE_URL = "http://vos2.asu.cas.cz/lamost_dr3/q/sdl/dlget"


def parse_metadata(metadata_file):
    metadata = parse(metadata_file)
    return metadata.get_first_table().to_table().to_pandas()["intensities"].iloc[0]


def parse_labeled_line(line, metadata, has_class):
    line_elements = [num.strip(" ") for num in line.split(",")]
    name = line_elements[0]
    if has_class:
        numbers = [float(num) for num in line_elements[1:-1]]
        header = list(metadata) + ["label"]
        numbers.append(int(line_elements[-1]))
    else:
        numbers = [float(num) for num in line_elements[1:]]
        header = list(metadata)
    return pd.DataFrame(data=[numbers], columns=header, index=[name])

def parse_args(argv):
    parser = argparse.ArgumentParser("vocloud_spark_preprocess")
    parser.add_argument("config", type=str)
    return parser.parse_args(argv)

def download_spectra_from(full_path):
    name, _ = os.path.splitext(os.path.basename(full_path))
    params_dict = {"FLUXCALIB": "normalized",
              "LAMBDA_MIN": "6255e-10",
              "LAMBDA_MAX": "6800e-10",
              "FORMAT": "application/x-votable+xml;serialization=tabledata",
              "ID": "ivo://asu.cas.cz/stel/lamost_dr3/{0}".format(name)}
    params = urlencode(params_dict)

    url = "{0}?{1}".format(BASE_URL, params)
    data = requests.get(url)
    return ("{0}.vot".format(os.path.splitext(name)[0]), data.text)

#def plot_spectra(spectra_file, header, out_folder):
#    with open(spectra_file) as spectra:
#        for line in spectra:
#            spectrum = parse_labeled_line(line, header)
#            spectrum.iloc[0].plot()
#            plt.savefig(out_folder + "/" + spectrum.index[0] + ".png")

def main(argv):
    logging.config.fileConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.ini"))
    parsed_args = parse_args(argv)
    spark_conf = SparkConf()
    sc = SparkContext(conf=spark_conf)
    with open(parsed_args.config) as in_config:
        preprocess_conf = json.load(in_config)
    source = preprocess_conf["input"]
    if not os.path.isdir(source):
        names = sc.textFile(source)
        files = names.map(download_spectra_from)
    else:
        files = sc.wholeTextFiles(source)
    metadata = parse_metadata(preprocess_conf["labeled"]["metadata"])
    labeled = sc.textFile(preprocess_conf["labeled"]["file"]).map(lambda x: parse_labeled_line(x, metadata, True)).cache()
    resampled = prep.preprocess(files, labeled).cache()
    header = resampled.take(1)[0].columns
    resampled.map(lambda x: x.to_csv(None, header=None).rstrip("\n")).coalesce(1).saveAsTextFile("out")
    os.rename("out/part-00000", preprocess_conf["output"])
#    if preprocess_conf["plot"]:
#        os.mkdir("plot")
#        plot_spectra(preprocess_conf["output"], header)


if __name__ == '__main__':
    main(sys.argv[1:])
