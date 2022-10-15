# Reconstructed_Radiograph_Tools
 
Hello and welcome to my honours project for Curtin University Bachelor of Radiation Science (Medical Imaging).

This repository contains a set of tools for manipulating CT data with Python. It was written in Jupyter Notebook using Anaconda Navigator to manage dependencies.


#### perspectiveless_DRR.ipynb ####
This Notebook can import a DICOM directory as a 3D numpy array, and then creates an average intensity projection over the entire dataset. The result closely resembles an x-ray but does not account for geometric factors such as beam divergence or source image distance. The Notebook contains some examples where the performance of this algorithm is timed, and some examples to demonstrate the same process in the sagittal and coronal plane.
