# KISN-PyLab
### Data processing pipeline for ephys and 3D tracking experiments
---------------------------------------------------------------------------------
This repository should facilitate the data processing pipeline for ephys, IMU and 3D tracking experiments.

It is useful for:
1. concatenating multiple Neuropixel recording sessions
2. running kilosort through Python
3. splitting clusters back to original sessions / getting spike times
4. reading sync events from the Neuropixel recording files
5. assessing whether the recording, tracking and IMU systems are synced
6. converting raw tracking files to a format loadable by the lab GUI

Please refer to the executor notebook for further instructions.


### Installation
---------------------------------------------------------------------------------
##### Prerequisites
Anaconda [(Python 3.7 version)](https://www.anaconda.com/distribution/#download-section)

##### Anaconda
To install KISN-PyLab on Windows, you can follow the steps below.
1. Open the command prompt. To do so click on the Windows icon and type "cmd" and press enter.
2. Type in the following into your new command prompt window:
~~~bash
conda create -n lab python=3.7
~~~
where "lab" can be any name you like. This will be the name of your environment inside which you'll install all relevant Python packages. Confirm with 'Y' that you want to create this environment.
3. Activate the newly created environment to start working with it by running:
~~~bash
activate lab
~~~
4. Install KISN-PyLab by running the following command in the virtual environment:
~~~bash
conda install -c bartulem kisn_pylab
~~~
This should work on most platforms. Let me know if it doesn't. 

##### Install from github
If you have `git`, you can simply clone the KISN-PyLab repository with `git clone https://github.com/bartulem/KISN-PyLab.git`. Otherwise, you can download the compressed file to your computer, depending on the OS. 
