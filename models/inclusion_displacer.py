from scipy.signal import windows
import numpy as np
import matplotlib.pyplot as plt

class InclusionDisplacer:
    
    def __init__(self, original_model, water_inclusion_pos):
        self.original_model = original_model
        self.water_inclusion_pos = water_inclusion_pos
        self.displaced_model = None  # This will store the model after displacement


    def displace_inclusions(self, water_inclusion_pos, lambda_val, alpha):
        """
        Displace the inclusions based on the given MATLAB code.
        
        Parameters:
        - xpos: Original x positions of the inclusions.
        - ypos: Original y positions of the inclusions.
        - lambda_val: Displacement parameter.
        - alpha: Gaussian window parameter.
        
        Returns:
        - new_xpos: Displaced x positions.
        - new_ypos: Displaced y positions.
        """

        new_water_inclusion_pos = water_inclusion_pos.copy()
        xpos = water_inclusion_pos[:, 0]
        zpos = water_inclusion_pos[:, 1]

        new_xpos = xpos + lambda_val * windows.gaussian(len(xpos), std=alpha)
        yvecFunc = lambda_val * windows.gaussian(2 * len(zpos), std=alpha)
        new_zpos = zpos - yvecFunc[:len(yvecFunc)//2]

        new_water_inclusion_pos[:, 0] = new_xpos
        new_water_inclusion_pos[:, 1] = new_zpos
        
        return new_water_inclusion_pos

    def apply_inclusions(self, new_water_inclusion_pos):

        """
        Apply the water inclusions to the model

        Parameters:
        self (SimulationModel): the model to apply the inclusions to
        water_inclusion_pos (np.array): the water inclusion positions

        Returns:
        None
        """

        nx = self.displaced_model.shape[0]
        nz = self.displaced_model.shape[2]

        # Create a mesh grid based on the model dimensions
        cols, rows = np.meshgrid(np.arange(nx), np.arange(nz))
        x_center_pixel = new_water_inclusion_pos[:, 0]
        z_center_pixel = new_water_inclusion_pos[:, 1]
        radius = new_water_inclusion_pos[:, 2]

        for i in range(len(new_water_inclusion_pos)):
            water_matrix = np.logical_or(water_matrix, ((rows - z_center_pixel) ** 2 + (cols - x_center_pixel) ** 2 <= (radius / self.discrete[2]) ** 2))

        # Update the model to include the water inclusions
        for i in range(nz):
            self.displaced_model[water_matrix[i], :, i] = 3  # Assuming that water is represented by the value 3 in the model

        self.displaced_model[:, :, 0:round(5.0/self.discrete[2])] = 0 # Freespace = 0
        self.displaced_model[:, :, round(105.0/self.discrete[2]):nz] = 2 # Bedrock = 2

    def displace(self, lambda_val, alpha):
        """
        Displace the inclusions in the model and store the result in self.displaced_model.
        
        Parameters:
        - lambda_val: Displacement parameter.
        - alpha: Gaussian window parameter.
        """
        new_water_inclusion_pos = self.displace_inclusions(self.water_inclusion_pos, lambda_val, alpha)

        # Create a new model similar to the original and apply the new positions.
        # You may need to write a method similar to water_inclusion to create this model.
        # For now, I'm assuming a placeholder method named "apply_inclusions"
        self.displaced_model = self.original_model
        self.apply_inclusions(new_water_inclusion_pos)
    

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

        X, Y = np.meshgrid(np.arange(0, self.x_size, self.discrete[0]), 
                        np.arange(0, self.z_size, self.discrete[2]))
        
        plt.pcolormesh(X, Y, self.model[:, round(transceiver[1]*self.discrete[1]), :].T)
        plt.scatter(transceiver[0], transceiver[2])
        plt.scatter(receiver[0], receiver[2])
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.clim(0, 3)
        plt.ylabel('depth [m]')
        plt.xlabel('distance [m]')
        plt.title(self.name)
        plt.savefig(self.path+'/figures/'+self.name+'.png', dpi=300)
        plt.close() 