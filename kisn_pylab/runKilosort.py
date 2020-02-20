# -*- coding: utf-8 -*-

"""

@author: bartulem

Run kilosort.

"""

import time
import matlab.engine
import os
import sys


def runKilo(fileDIR, kilosortDIR):

    fileDIR = fileDIR
    kilosortDIR = kilosortDIR
    
    # test that the dir is there
    if(not os.path.exists(fileDIR)):
        print('Could not find directory {}, try again.'.format(fileDIR))
        sys.exit()
    
    print('Kilosort to be run on file: {}.'.format(fileDIR))
    
    # run kilosort
    print('Running kilosort, please be patient - this could take 5-10 min.')
    t = time.time()
    eng = matlab.engine.start_matlab()
    eng.cd(kilosortDIR, nargout=0)
    eng.ls(nargout=0)
    eng.master_kilosort(fileDIR, nargout=0)
    eng.quit()
    print('Running kilosort took {:.2f} seconds.\n'.format(time.time() - t))
