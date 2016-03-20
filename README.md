# Import of votable spectra for VO-CLOUD


## Requirements:
* Python packages are listed in `requirements.txt`
* To run, you need to have Spark installed on your machine and have its' Python bindings
included in your PYTHONPATH
* Python 3.5, support for Python 2.7 is in planned (no E.T.A)

## Installation:
Setup Spark on your machine and run
``export PYTHONPATH=$SPARK_HOME/python``. If you're using a virtual environment,
you can symlink the path in the virtual envs site-packages dir.

```
pip install -r requirements.txt
pip install -e .
```
# Running

To run on your local machine (e. g. for testing) execute `spark-submit --name "test" --master
"local" PACKAGE_HOME/bin/vocloud_preprocess.py <CONFIG_FILE>`
