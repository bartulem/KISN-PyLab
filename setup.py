import os
from setuptools import setup


# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="kisn_pylab",
    version="3.0.0",
    author="bartulem",
    author_email="mimica.bartul@gmail.com",
    description=("For running e-phys/3D tracking experiments"),
    license="LICENSE",
    keywords="ephys tracking imu",
    url="https://github.com/bartulem/KISN-PyLab",
    packages=['kisn_pylab'],
    long_description=read('README.md'),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ], install_requires=['pandas', 'numpy', 'tqdm', 'scipy', 'matplotlib', 'scikit-learn', 'plotly', 'astropy']
)
