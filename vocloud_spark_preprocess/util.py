__author__ = 'Andrej Palicka <andrej.palicka@merck.com>'
import sys
import os
import zipfile

def add_dependencies():
    dep_dir = os.path.expanduser("~") + "/dependencies"
    if not os.path.isdir(dep_dir):
        os.mkdir(dep_dir)
        dep_zip_list = [i for i, val in enumerate(sys.path) if val.endswith("dependencies.zip")]
        if len(dep_zip_list) > 0:
            dep_zip = sys.path[dep_zip_list[0]]
            with open(dep_zip, 'rb') as dep_zip_file:
                z = zipfile.ZipFile(dep_zip_file)
                for name in z.namelist():
                    z.extract(name, dep_dir)
    sys.path.insert(0, dep_dir)
