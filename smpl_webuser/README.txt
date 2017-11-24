License:
========
To learn about SMPL, please visit our website: http://smpl.is.tue.mpg
You can find the SMPL paper at: http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf

Visit our downloads page to download some sample animation files (FBX), and python code:
http://smpl.is.tue.mpg/downloads

For comments or questions, please email us at: smpl@tuebingen.mpg.de


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy 		 [https://github.com/mattloper/chumpy]
- OpenCV 		 [http://opencv.org/downloads.html] 


Getting Started:
================

1. Extract the Code:
--------------------
Extract the 'smpl.zip' file to your home directory (or any other location you wish)


2. Set the PYTHONPATH:
----------------------
We need to update the PYTHONPATH environment variable so that the system knows how to find the SMPL code. Add the following lines to your ~/.bash_profile file (create it if it doesn't exist; Linux users might have ~/.bashrc file instead), replacing ~/smpl with the location where you extracted the smpl.zip file:

	SMPL_LOCATION=~/smpl
	export PYTHONPATH=$PYTHONPATH:$SMPL_LOCATION


Open a new terminal window to check if the python path has been updated by typing the following:
>  echo $PYTHONPATH


3. Run the Hello World scripts:
-------------------------------
In the new Terminal window, navigate to the smpl/smpl_webuser/hello_world directory. You can run the hello world scripts now by typing the following:

> python hello_smpl.py

OR 

> python render_smpl.py



Note:
Both of these scripts will require the dependencies listed above. The scripts are provided as a sample to help you get started. 

