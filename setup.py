import os
from setuptools import setup
import tf_ops


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Read metadata from version file
classifiers = [
    "Programming Language :: Python :: 3",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Topic :: Software Development :: Libraries",
    "Topic :: Utilities",
]

setup(
    name='tf_ops',
    version=tf_ops.__version__,
    author="Fergal Cotter",
    author_email="fbc23@cam.ac.uk",
    description=("Convenience Functions for Tensorflow"),
    license="MIT",
    keywords="tensorflow, complex convolution",
    url="https://github.com/fbcotter/tf_ops.git",
    long_description=read('README.rst'),
    classifiers=classifiers,
    command_options={
        'build_sphinx': {
            'source-dir': 'docs/',
            'build-dir': 'build/docs',
            'project': ('setup.py', 'tf_ops'),
            'version': ('setup.py', tf_ops.__version__)}},
    tests_require=["pytest"],
    py_modules=["tf_ops"],
    install_requires=['numpy', 'six',],
)
