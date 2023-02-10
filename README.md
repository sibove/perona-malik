# Perona-Malik diffusion

## Disclaimer
This repository contains the code I used in my final project for a Partial Differential Equations course during my exchange semester at [Syddansk Universitet](https://www.sdu.dk) 2021. I have not updated the code since, so I am not aware if newer versions of the libraries involved broke the scripts. 

[Perona-Malik diffusion](https://en.wikipedia.org/wiki/Anisotropic_diffusion) is a technique developed to reduce image noise that focuses on preserving the edges of the various shapes present in the image. 

## The idea
An intuitive idea when attempting to reduce noise in an image is to use some kind of _local mean_ of the RGB values at each pixel of the image. One could arbitrarily choose to compute said mean over a $3\times 3$ or $5\times 5$ square centered at the pixel, and would quickly notice how the arithmetic mean is not a good way to go, since every pixel in the square over which the average is calculated is given the same weight in determining the color of the central pixel. 
It is then natural to resort to a weighted average instead, in which pixels closer to the center of the "averaging square" have a bigger weight and pixels closer to the border have a low weight in the average. A possible choice for assigning the weights this way is to use a (bivariate) Gaussian distribution centered on the pixel for which we are calculating the denoised color. This choice allows us to compute the average over the entire domain as well, instead of restricting it to a square like before. 

We can now explain the link between image denoising and PDEs, restricting ourselves to grayscale (just for the sake of simplicity - for RGB images we can just split the three channels). 
An $H\times L$ bitmap image can be represented as a function 
$$I: \{0, \dots, L-1\}\times \{0, \dots, H-1\}\to \{0, 255\}$$.
We can embed these in the set of the (smooth) functions of the form
$$I: [0, L-1]\times [0, H-1]\to [0,255]$$

Computing the "Gaussian-weighted" mean we described above can be interpreted as applying a [convolution](https://en.wikipedia.org/wiki/Convolution) of the function $I$ with the Gaussian density function, which we will denote $\Phi$, and then projecting the result back to the discrete space of bitmap images. 
Mathematically speaking, convolution over a domain $\Omega$ (in our case $[0, L-1] \times [0,H-1]$) is defined as 
$$(\Phi * I)(x)=\int_\Omega \Phi(x-y)I(y)dy.$$

It turns out, convolving with a Gaussian density is the same thing as solving the heat equation
$$I_t = \nabla\cdot(D\nabla I)$$
in time, with a constant diffusion coefficient $D$ which is related to the variance of the Gaussian we convolute the starting image with. Selecting a constant coefficient, however, does not lead to good results.

![gaussian](https://user-images.githubusercontent.com/125075914/218161909-ac9bbd49-b68b-462f-80a9-c4ae6632fbcb.jpeg)

The problem is that the smoothing process has to be adaptive: we want smaller diffusion coefficients closer to the edges of the various shapes in the image, so that the edge itself or anything on the other side has a very small weight. 
The solution proposed by Perona and Malik was to estimate the diffusion parameter $D = D(x,y,t)$ using the gradient $\nabla I(x,y,t)$, or rather a function of its norm. Two proposed function families were
$$g(\|\nabla I\|) = e^{-(\|\nabla I\|/K)^2}$$
and 
$$g(\|\nabla I\|) = \frac{1}{1 + (\|\nabla I\|/K)^2},$$
with $K>0$ chosen either by hand or according to some noise estimate. 

## Numerical solution
We will solve the heat equation above by reformulating it as a variational problem. The first thing we have to do is to decide a time integration algorithm, and for its stability properties we used the [implicit Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method). This means that at the $n$-th time step, we need to solve the following problem for $u$: 
$$\frac{u^n - u^{n-1}}{\Delta t} = \nabla\cdot(D\nabla u^n),$$
where $\Delta t$ is the selected time-step and we remember that $D$ depends on time and space. 

Now we multiply by the _test function_ $\phi$ and integrate over the domain. We also choose to leave the boundary fixed to the initial value for all times, and applying this condition yields the following variational problem: 
$$\int_\Omega u^n\phi dxdy + \int_\Omega \Delta t D \nabla u^n \cdot \nabla\phi dxdy = \int_\Omega u^{n-1}\phi dxdy\quad\forall \phi.$$

To solve this, we use the [FEniCS](https://fenicsproject.org) library for Python. The code is included in [peronamalik.py](peronamalik.py). A mean squared error script is included in (quant_error.py)[quant_error.py]. It was used to test the main script by comparing the original image to the result of adding artificial noise and then denoising the image with the anisotropic technique. 
