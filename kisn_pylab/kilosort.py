# -*- coding: utf-8 -*-

"""

@author: bartulem

Run Kilosort2 through Python.

As it stands (spring/summer 2020), to use Kilosort2 one still requires Matlab. To ensure it works,
one needs a specific combination of Matlab, the GPU driver version and CUDA compiler files.
On the lab computer, I set it up to work on Matlab R2019b, driver version 10.2. (GeForce RTX 2080 Ti)
and v10.1 CUDA. !!! NB: a different Matlab or driver version would require different CUDA files !!!
Additionally, since I don't change the config file from session to session, I wrote the script
below to run Matlab code through Python, such that the whole processing pipeline would remain Pythonic.
Apart from Matlab, in order for this to run, you need to install the "matlab engine"; further instructions
can be found here: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
However, if you need to modify the config file, you either need to change the code below accordingly
or go to Matlab and run Kilosort2 the old school way.

"""

import time
import matlab.engine
import os
import sys


def run_kilosort(file_dir, kilosort2_dir):
    
    # check that the data directory is there
    if not os.path.exists(file_dir):
        print('Could not find data directory {}, try again.'.format(file_dir))
        sys.exit()

    # check that the Kilosort2 directory is there
    if not os.path.exists(kilosort2_dir):
        print('Could not find Kilosort directory {}, try again.'.format(kilosort2_dir))
        sys.exit()
    
    print('Kilosort2 to be run on file: {}.'.format(file_dir))
    
    # run Kilosort2
    print('Running Kilosort2, please be patient - this could take >1 hour.')

    t = time.time()
    eng = matlab.engine.start_matlab()
    eng.cd(kilosort2_dir, nargout=0)
    eng.ls(nargout=0)
    eng.master_kilosort(file_dir, nargout=0)
    eng.quit()

    print('Finished! Running Kilosort2 took {:.2f} minutes.\n'.format((time.time() - t) / 60))
