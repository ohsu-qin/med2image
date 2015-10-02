import os

import re
from setuptools import setup

def version():
    """
    Return package version as listed in the `__init.py__` `__version__`
    variable.
    """
    init = open(os.path.join(os.getcwd(), '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init).group(1)


def requires():
    """
    @return: the ``requirements.txt`` package specifications
    """
    with open('requirements.txt') as f:
       return f.read().splitlines()


setup(
    name = 'med2image',
    version = version(),
    author = 'Fetal Neonatal Neuroimaging and Developmental Science Center',
    platforms = 'Any',
    license = 'MIT',
    keywords = 'Medical Imaging DICOM NiFTI JPEG',
    packages = ['med2image'],
    package_dir = dict(med2image='..'),
    scripts = ['med2image.py'],
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
