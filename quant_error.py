"""
Script for comparing two images of the same size by calculating two types of difference (error).
Example call:   
quant_error.py  image1.jpg  image2.jpg
"""


import sys, PIL.Image, numpy

image1 = PIL.Image.open(sys.argv[1]).convert('L')
image2 = PIL.Image.open(sys.argv[2]).convert('L')

data1 = numpy.asarray(image1)
data2 = numpy.asarray(image2)

H, L = data1.shape

data1 = numpy.concatenate(data1)
data2 = numpy.concatenate(data2)

# Max error (L1 norm on the discretized domain)
max_error = numpy.amax(abs(data1 - data2))
print("The maximum error is %d" % max_error)

# Mean squared error
mse = sum((data1-data2)**2) / (H * L)
print("The mean squared error is %.2f" % mse)