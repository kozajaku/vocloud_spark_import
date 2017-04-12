import os
import sys
import argparse
from astropy.io.votable import parse
import pandas as pd

import vocloud_spark_preprocess.preprocess_data as prep
import logging
import logging.config
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
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
        label = int(line_elements[-1])
        numbers.append(label)
    else:
        numbers = [float(num) for num in line_elements[1:]]
        header = list(metadata)
    return pd.DataFrame(data=[numbers], columns=header, index=[name])


def parse_args(argv):
    parser = argparse.ArgumentParser("vocloud_spark_preprocess")
    parser.add_argument("config", type=str)
    return parser.parse_args(argv)


# def plot_spectra(spectra_file, header, out_folder):
#    with open(spectra_file) as spectra:
#        for line in spectra:
#            spectrum = parse_labeled_line(line, header)
#            spectrum.iloc[0].plot()
#            plt.savefig(out_folder + "/" + spectrum.index[0] + ".png")


def transform_labels(df):
    copy = df.copy()
    if copy.iloc[0]["label"] == 2:
        return copy.set_value(df.iloc[0].name, "label", 0)
    else:
        return copy.set_value(df.iloc[0].name, "label", 1)


def main(argv):
    logging.config.fileConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.ini"))
    parsed_args = parse_args(argv)
    spark_conf = SparkConf()
    sc = SparkContext(conf=spark_conf)
    with open(parsed_args.config) as in_config:
        preprocess_conf = json.load(in_config)
    if preprocess_conf.get("avro", False):
        # binary files saved inside avro format
        sql_context = SQLContext(sc)  # requires parameter --packages com.databricks:spark-avro_2.10:2.0.1
        input = preprocess_conf["input"]
        if not input.endswith("*.avro"):
            input = os.path.join(input, "*.avro")
        df = sql_context.read.format("com.databricks.spark.avro").load(input)
        files = df.rdd
    else:
        if preprocess_conf.get("binary_input", True):
            files = sc.binaryFiles(preprocess_conf["input"], preprocess_conf.get('partitions', 4000))
        else:
            files = sc.wholeTextFiles(preprocess_conf["input"], preprocess_conf.get('partitions', 4000))
    files = files.repartition(preprocess_conf.get('partitions', 4000))
    metadata = parse_metadata(preprocess_conf["labeled"]["metadata"])
    labeled = sc.textFile(preprocess_conf["labeled"]["file"], preprocess_conf.get('partitions', 4000)). \
        map(lambda x: parse_labeled_line(x, metadata, True)).filter(lambda x: x.iloc[0]["label"] != 4).map(
        transform_labels)
    header, resampled = prep.preprocess(sc, files, labeled, label=preprocess_conf.get('label', True),
                                        cut=preprocess_conf.get("cut", {"low": 6300, "high": 6700}),
                                        pca=preprocess_conf.get("pca", None),
                                        partitions=preprocess_conf.get('partitions', 100))
    resampled.map(lambda x: x.to_csv(None, header=None).rstrip("\n")).saveAsTextFile(preprocess_conf["output"])
    # os.rename("out/part-00000", preprocess_conf["output"])


#    if preprocess_conf["plot"]:
#        os.mkdir("plot")
#        plot_spectra(preprocess_conf["output"], header)


if __name__ == '__main__':
    main(sys.argv[1:])
