
import numpy as np
import matplotlib.pyplot as plt


try:

     phi_data = np.loadtxt('data/phi_solution.txt')

     ex_data = np.loadtxt('data/Ex_field.txt')

     ey_data = np.loadtxt('data/Ey_field.txt')

except IOError:

     print("Error: Make sure the solver has run and generated files in the 'data/' directory.")
 
     exit()



# Plot Electric Potential (phi)

plt.figure(figsize=(8, 6))

plt.imshow(phi_data, cmap='viridis', origin='lower')

plt.colorbar(label='Electric Potential (phi)')

plt.title('Electric Potential Field')

plt.xlabel('X-grid index')

plt.ylabel('Y-grid index')

plt.tight_layout()

plt.show()



# Plot Electric Field (E) as a quiver plot

# Subsample for clearer visualization on large grids

step = max(1, phi_data.shape[0] // 20) # Adjust subsampling step

Y, X = np.mgrid[0:phi_data.shape[0]:step, 0:phi_data.shape[1]:step]



plt.figure(figsize=(10, 8))

plt.imshow(phi_data, cmap='viridis', origin='lower', alpha=0.6) # Background potential

plt.colorbar(label='Electric Potential (phi)')

plt.quiver(X, Y, ex_data[::step, ::step], ey_data[::step, ::step], color='white', scale=np.max(np.sqrt(ex_data**2 + ey_data**2))*2, width=0.002)

plt.title('Electric Field (Vector Plot)')

plt.xlabel('X-grid index')

plt.ylabel('Y-grid index')

plt.tight_layout()

plt.show()



