# KISN-PyLab (v3.0.0)
### Data processing pipeline for e-phys and 3D tracking experiments
---------------------------------------------------------------------------------
This repository should facilitate the data processing pipeline for e-phys, IMU and 3D tracking experiments.

![system diagram](./system_diagram.png)

It is useful, in single- or dual-probe settings, for:
1. concatenating multiple Neuropixel recording sessions
2. running Kilosort2 through Python
3. computing cluster quality measures, and separating single units and MUA
4. locating the surface recording channel
5. reading random sync events from the Neuropixel recording, tracking and IMU files
6. assessing whether the recording, tracking and IMU systems are synced
7. splitting clusters back to original sessions / getting spike times
8. converting raw tracking files to a format loadable by the lab GUI
9. re-heading different analyses (.mat) files based on one template session

Please refer to the **executor notebook** for further instructions.

This pipeline was used to process data in "Behavioral decomposition reveals rich encoding structure employed across neocortex" ([bioRxiv](https://www.biorxiv.org/content/10.1101/2022.02.08.479515v1), 2022).

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
Confirm with 'Y' that you want to install this package. This should work on 64-bit platforms. Let me know if it doesn't. 

##### Github
If you have `git`, you can simply clone the KISN-PyLab repository with `git clone https://github.com/bartulem/KISN-PyLab.git`. Otherwise, you can download the compressed file to your computer, depending on the OS. 

After downloading KISN-PyLab, to install, run:

~~~bash
python setup.py install
~~~

If you are not root, run:

~~~bash
python setup.py install --user
~~~
