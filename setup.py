import os

import re
from setuptools import (find_packages, setup)


def version(package):
    """
    Return package version as listed in the `__init.py__` `__version__`
    variable.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


def requires():
    """
    @return: the ``requirements.txt`` package specifications
    """
    with open('requirements.txt') as f:
       return f.read().splitlines()


setup(
    name = 'med2image',
    version = version('med2image'),
    author = 'Fetal Neonatal Neuroimaging and Developmental Science Center',
    platforms = 'Any',
    license = 'MIT',
    keywords = 'Medical Imaging DICOM NiFTI JPEG',
    packages = find_packages(),
    scripts = ['bin/med2image'],
    url = 'https://github.com/FNNDSC/med2image',
    description = 'Converts medical images to more displayable formats',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires = requires()
)
