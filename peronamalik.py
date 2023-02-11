"""
Script for Perona-Malik anisotropic diffusion.
The image to be denoised shall be renamed to noisy.jpg.
"""

from fenics import *
import sys, PIL, numpy, time
import matplotlib.pyplot as plt

# Clock starts
start = time.time()

# Import noisy image and convert it to grayscale if necessary
image = PIL.Image.open('noisy.jpg').convert('L')

# Convert the image to an array for manipulation
data = numpy.asarray(image)
#print(data.shape) # Prints image dimensions

# Create mesh and define function space
# Rectangular mesh with a vertex in each pixel position
mesh = RectangleMesh(Point(0, 0), Point(data.shape[0]-1,\
 data.shape[1]-1), data.shape[0]-1, data.shape[1]-1)
V = FunctionSpace(mesh, 'Lagrange', 1)

## Initial and boundary conditions
# Initial condition is the intensity array from the starting image
class InitialCondition(UserExpression):
    def eval(self, values, x):
        values[0] = data[int(x[0])][int(x[1])] 
    def value_shape(self):
        return()

u0 = InitialCondition(degree = 2)
u0 = interpolate(u0, V) 

# On the boundary we fix the initial condition for all times.
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u0, boundary)

## Problem solution, using implicit Euler time stepping.
# Parameters
t0 = 0.0
T = 20.0
dt = 1.0
# Anisotropic diffusion coefficient 
K = 10.0 # Should actually be K^2 in the following, but we choose it positive so it does not matter
D = 50.0 * 1/(1 + inner(grad(u0),grad(u0))/K) 
#D = Constant(50.0) # Isotropic diffusion

t = t0
u_ = u0 #store the initial condition in previous time step

# Define variational problem 
u = TrialFunction(V)
v = TestFunction(V)
a = u*v*dx + dt*D*inner(grad(u),grad(v))*dx # Implicit Euler time-stepping
l = u_*v*dx 

# File to store the solution
output = File('work/results/paraview/output.pvd')

# Initialize solution
u = Function(V)

# Assemble the matrix A 
A = assemble(a)

# Apply the boundary conditions for the matrix
bc.apply(A)

# Time stepping
while t <= T:
    # Compute the solution at time t
    L = assemble(l)
    bc.apply(L)
    solve(A, u.vector(), L) 
    """
    # Save solution as image
    u_temp = numpy.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            u_temp[i][j] = u(i,j)

    u_image = PIL.Image.fromarray(u_temp)
    u_image = u_image.convert("L")
    u_image.save("work/results/images/u_t" + str(round(t,3)) + ".jpg")
    """
    # Update and save paraview file
    u_.assign(u)
    t += dt
    output << u
    
    # Display the current time
    print('Processed time = %5.2f' %(t-dt))

# Clock stops
end = time.time()
print('Elapsed time = %d seconds' %(end - start))

# Parallel computing splits the domain!    
#plot(u)
#plt.show()

image = PIL.Image.fromarray(data)
image.save("grayscale.jpg")

# Metadata attached to the Paraview files
with open("work/results/paraview/metadata.txt", "w") as meta:
    meta.write("Elaboration of image %s. \n"\
    "Diffusion calculated for time ranging from %.3f to %.3f, "\
    "with steps of size %.3f." % ("verynoisytree", t0, T, dt))
    

# Save final iteration as image
u_final = numpy.zeros((data.shape[0], data.shape[1]))
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        u_final[i][j] = u(i,j)

u_image = PIL.Image.fromarray(u_final)
u_image = u_image.convert("L")
u_image.save("denoised.jpg")


