from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.3'
DESCRIPTION = 'A package made for streamlining Variational Autoencoders'
LONG_DESCRIPTION = 'A package that lets you run Variational Autoencoders on any dataset that you would like, in an easy and streamlined manner, you should however learn how Variational Autoencoders work before using this package.'

# Setting up
setup(
    name="autovariate",
    version=VERSION,
    author="Siris2314 (Arihant Tripathi)",
    author_email="tarihant2001@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['torch', 'tqdm', 'torchvision', 'py-cpuinfo', 'py_console'], # add any additional packages that are nedded
    keywords=['python', 'NLP', 'Natural Language Processing', 'Deep Learning', 'Machine Learning', 'Variational Autoencoders', 'Autoencoders', 'AutoVariate'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)