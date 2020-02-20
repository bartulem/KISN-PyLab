import os
from setuptools import setup, find_packages

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "kisn-pylab",
    version = "1.0.0",
    author = "bartulem",
    author_email = "mimica.bartul@gmail.com",
    description = ("For running lab experiments"),
    license = "BSD",
    keywords = "ephys tracking imu",
    url = "https://github.com/bartulem/KISN-PyLab",
    packages=find_packages()
    long_description=read('README'),
    classifiers=[
        "License :: OSI Approved :: BSD License",
    ],
)
