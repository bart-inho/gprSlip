from scipy.signal import windows
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class InclusionDisplacer:
    
    def __init__(self, original_model, water_inclusion_pos):
        self.original_model = np.copy(original_model.model)
        self.water_inclusion_pos = np.copy(water_inclusion_pos)
        self.new_water_inclusion_pos = np.copy(water_inclusion_pos)

        self.dx = original_model.discrete[0]
        self.dy = original_model.discrete[1]
        self.dz = original_model.discrete[2]

        self.x_size = original_model.x_size
        self.y_size = original_model.y_size
        self.z_size = original_model.z_size

        self.name = original_model.name
        self.path = original_model.path

        self.displaced_model = np.zeros(self.original_model.shape)

    def gausswin(self, N, alpha, shift=0):
        """ Gaussian window function with an optional shift. """
        n = np.arange(0, N) - (N - 1) / 2 - shift
        w = np.exp(-(1/2) * (alpha * n / ((N - 1) / 2))**2)
        return w
    
    def exponential_field(self, X, Z, alpha, x0=None, z0=None):
        """ Exponential field function. """
        if x0 is None:
            x0 = self.x_size / 2
        if z0 is None:
            z0 = self.z_size

        return np.exp(-alpha * np.sqrt((X - x0)**2 + (Z - z0)**2))

    def displace_inclusions(self, lambda_val, alpha):
        """ Displace the inclusions based on a repulsive Gaussian field. """
        
        nx = int(self.x_size / self.dx)
        nz = int(self.z_size / self.dz) 
            
        # Creating the 2D Gaussian hill in the middle of the domain
        gaussian_hill = self.gausswin(nx, alpha)[:, np.newaxis] * self.gausswin(nz, alpha, shift=nz/2)

        # plt.imshow(gaussian_hill.T)
        # plt.title("Gaussian hill")
        # plt.colorbar()
        # plt.xlabel("nx")
        # plt.ylabel("nz")
        # plt.show()

        # Compute the gradients of the Gaussian hill, these give us the direction of the "push"
        gx, gz = np.gradient(gaussian_hill)
            
        for idx, (x, z, radius) in enumerate(self.water_inclusion_pos):
            # Find the nearest grid index
            ix, iz = int(x), int(z)  # Adjusted to consider grid spacing
                
            # Displacement proportional to gradient (i.e., slope) and scaled by lambda_val
            disp_x = -lambda_val * gx[ix, iz]  # Negative because we want repulsion
            disp_z = -lambda_val * gz[ix, iz]

            # Update positions
            new_x = x + disp_x
            new_z = z + disp_z

            self.new_water_inclusion_pos[idx] = [new_x, new_z, radius]

    def apply_inclusions(self):
        nx = int(self.x_size / self.dz)
        ny = int(self.y_size / self.dy)
        nz = int(self.z_size / self.dx)

        # Initialize the model
        self.displaced_model = np.zeros((nx, ny, nz))
        self.displaced_model[:, :, round(5.0 / self.dz):nz] = 1 # Glacier = 1
        
        # Create a grid for i and j
        i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')

        for inclusion in tqdm(self.new_water_inclusion_pos, desc="Processing inclusions"):
            # Calculate the squared distance for all points at once
            mask = ((j - inclusion[1]) ** 2 + (i - inclusion[0]) ** 2) <= ((inclusion[2] / self.dz) ** 2)
            
            # Reshape the mask
            mask_reshaped = mask[:, np.newaxis, :]

            # Use the reshaped mask to update the displaced_model
            self.displaced_model[mask_reshaped] = 3

        self.displaced_model[:, :, 0:round(5.0/self.dz)] = 0 # Freespace = 0
        self.displaced_model[:, :, round(105.0/self.dz):nz] = 2 # Bedrock = 2

    def plot_displacement(self):
        """
        Plot the displacement of the inclusions.

        Parameters:
        None

        Returns:
        None
        """

        plt.figure(figsize=(10, 10))
        plt.plot(self.water_inclusion_pos[:, 0]*self.dx, self.water_inclusion_pos[:, 1]*self.dy, 'o')
        plt.plot(self.new_water_inclusion_pos[:, 0]*self.dx, self.new_water_inclusion_pos[:, 1]*self.dy, 'o')
        plt.plot(self.x_size/2, self.z_size, 'x')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.xlim(0, self.x_size)
        plt.ylim(self.z_size, 0)
        plt.ylabel('depth [ ]')
        plt.xlabel('distance [ ]')
        plt.legend(['Original', 'Displaced'])
        plt.title(self.name + ' displacement')
        plt.savefig(self.path+'/figures/'+self.name+'_displacement.png')
        plt.close()

    def displace(self, lambda_val=3e6, alpha=3):
        """
        Displace the inclusions in the model and store the result in self.displaced_model.
        
        Parameters:
        - lambda_val: Displacement parameter.
        - alpha: Gaussian window parameter.
        """
        self.displace_inclusions(lambda_val, alpha)
        self.apply_inclusions() 
        self.plot_displacement()   

    def plot_displaced_model(self, transceiver, receiver):
        """
        Plot the initial model

        Parameters:
        self (SimulationModel): the model to plot
        transceiver (np.array): the transceiver position
        receiver (np.array): the receiver position

        Returns:
        None
        """

        X, Y = np.meshgrid(np.arange(0, self.x_size, self.dx), 
                        np.arange(0, self.z_size, self.dz))
        
        plt.pcolormesh(X, Y, self.displaced_model[:, round(transceiver[1]*self.dy), :].T)
        plt.scatter(transceiver[0], transceiver[2])
        plt.scatter(receiver[0], receiver[2])
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.clim(0, 3)
        plt.ylabel('depth [m]')
        plt.xlabel('distance [m]')
        plt.title(self.name + ' displaced')
        plt.savefig(self.path+'/figures/'+self.name+'_displaced.png')
        plt.close() 