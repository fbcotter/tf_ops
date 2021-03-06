import os
from setuptools import setup, find_packages
# Imports the __version__ variable
#  exec(open(os.path.join(os.path.dirname(__file__), 'version.py')).read())


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def readlines(fname):
    enc = 'utf-8'
    return open(os.path.join(os.path.dirname(__file__), fname), 'r',
                encoding=enc).readlines()


def read(fname):
    enc = 'utf-8'
    return open(os.path.join(os.path.dirname(__file__), fname), 'r',
                encoding=enc).read()


# Read metadata from version file
def get_version():
    f = readlines("tf_ops/__init__.py")
    for line in f:
        if line.startswith("__version__"):
            return line[15:-2]
    raise Exception("Could not find version number")


# Read metadata from version file
classifiers = [
    "Programming Language :: Python :: 3",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Topic :: Software Development :: Libraries",
    "Topic :: Utilities",
]

install_requires = [
    'numpy',
]

extras_require = {
    'docs': ['sphinx', 'docutils', 'matplotlib', 'ipython', ],
    'tf': ['tensorflow>=1.1.0'],
    'tf_gpu': ['tensorflow-gpu>=1.1.0'],
}

setup(
    name='tf_ops',
    version=get_version(),
    author="Fergal Cotter",
    author_email="fbc23@cam.ac.uk",
    description=("Convenience Functions for Tensorflow"),
    license="MIT",
    keywords="tensorflow, complex convolution",
    url="https://github.com/fbcotter/tf_ops.git",
    packages=find_packages(exclude=["tests.*", "tests"]),
    long_description=read('README.rst'),
    classifiers=classifiers,
    install_requires=install_requires,
    tests_require=["pytest"],
    extras_require=extras_require
)
