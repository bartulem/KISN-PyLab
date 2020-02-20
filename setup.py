import os
from setuptools import setup, find_packages

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "kisn_pylab",
    version = "1.0.0",
    author = "bartulem",
    author_email = "mimica.bartul@gmail.com",
    description = ("For running lab experiments"),
    license = "LICENSE",
    keywords = "ephys tracking imu",
    url = "https://github.com/bartulem/KISN-PyLab/kisn_pylab",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "License :: GNU License",
    ],
)
