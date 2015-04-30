# Parallel-Image-Analogies

# Windows Instructions
## Parallel Image Analogies only runs on Python 2.7 for now due to print compatibilities. TODO: fix it.
## Install the following dependencies using the command pip (remember to run as admin)
	pip install numpy matplotlib
Note that numpy will take a while since it will use the C++ compiler.
### Install Microsoft Visual Studio C++ Compiler for python in order to use numpy
	http://www.microsoft.com/en-us/download/details.aspx?id=44266
## Get a copy of the OpenCV and Scipy here and use pip install on the whl files (check if you are getting the right version)
	http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

#Code for CS205 Final Project

This repository includes code for both the serial and parallel implementations of the Image Analogy algorithm described in the research paper "Image Analogies" by Hertzmann, et al. (http://sites.fas.harvard.edu/~cs278/papers/analogies-72dpi.pdf).

It includes the following files:

* image_analogies_serial.py 

This is the serial implementation, which can be run with the following line:

	python image_analogies_serial.py [imageA_path] [imageA_prime_path] [imageB_path] [output_file]

* image_analogies_parallel.py

This is the parallel implementation, which utilizes Cheetah, CUDA, and MPI. It must be run on a cluster with GPU-equipped nodes, with the following command:

	mpiexec -n [num_processes] python image_analogies_mpi.py [imageA_path] [imageA_prime_path] [imageB_path] [output_file]

* run.qsub

This is a script that can be submitted to the headnode of a cluster, to run image_analogies_parallel.py on more than one node (i.e. if you need > 2 GPUs on the SEAS Resonance cluster). Use the following command:

	qsub run.qsub

To customize the input files and number of processes, you have to go into the qsub file and manually edit it.

* graph_maker.py, graph_maker2.py

Not relevant to the program execution, these are the scripts that we used to generate the graphs of our results.

We've also included sample image files that the code can be run on:

* artistic1_A1.jpg, artistic1_A2.jpg (artsy effects)
* embossA1.jpg, embossA2.jpg (emboss effects)
* blurA1.jpg, blurA2.jpg (blur effects)
* artistic1_B1.jpg, blurB1.jpg, fun.jpg (sample images to run the transformations on)


