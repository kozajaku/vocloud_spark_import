#!/usr/bin/env python3
import os
import sys
import argparse

import pandas as pd

import vocloud_spark_preprocess.preprocess_data as prep
import logging
import logging.config
from astropy.io.votable import parse
from pyspark import SparkConf, SparkContext
import json


__author__ = 'Andrej Palicka <andrej.palicka@merck.com>'

def parse_metadata(metadata_file):
    metadata = parse(metadata_file)
    return metadata.get_first_table().to_table().to_pandas()["intensities"].iloc[0]


def parse_labeled_line(line, metadata, has_class):
    line_elements = [num.strip(" ") for num in line.split(",")]
    name = line_elements[0]
    if has_class:
        numbers = [float(num) for num in line_elements[1:-1]]
        header = list(metadata) + ["label"]
        numbers.append(line_elements[-1])
    else:
        numbers = [float(num) for num in line_elements[1:]]
        header = list(metadata)
    return pd.DataFrame(data=[numbers], columns=header, index=[name])

def parse_args(argv):
    parser = argparse.ArgumentParser("vocloud_spark_preprocess")
    parser.add_argument("config", type=str)
    return parser.parse_args(argv)

def main(argv):
    logging.config.fileConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.ini"))
    parsed_args = parse_args(argv)
    spark_conf = SparkConf()
    sc = SparkContext(conf=spark_conf)
    preprocessor = prep.Preprocessing()
    with open(parsed_args.config) as in_config:
        preprocess_conf = json.load(in_config)
    files = sc.wholeTextFiles(preprocess_conf["input"])
    metadata = parse_metadata(preprocess_conf["labeled"]["metadata"])
    labeled = sc.textFile(preprocess_conf["labeled"]["file"]).map(lambda x: parse_labeled_line(x, metadata, True)).cache()
    resampled = preprocessor.preprocess(files, labeled).cache()
    resampled.map(lambda x: x.to_csv(None, header=None).rstrip("\n")).saveAsTextFile("out")

if __name__ == '__main__':
    main(sys.argv[1:])
