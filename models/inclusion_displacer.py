import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class InclusionDisplacer:
    
    def __init__(self, original_model, water_inclusion_pos):
        """
        Initialize the InclusionDisplacer class.

        Parameters:
        - original_model: The original model.
        - water_inclusion_pos: The position of the water inclusions.
        """

        self.original_model = np.copy(original_model.model)
        self.water_inclusion_pos = np.copy(water_inclusion_pos)
        self.new_water_inclusion_pos = np.copy(water_inclusion_pos)

        self.discrete = original_model.discrete

        self.dx = original_model.discrete[0]
        self.dy = original_model.discrete[1]
        self.dz = original_model.discrete[2]

        self.x_size = original_model.x_size
        self.y_size = original_model.y_size
        self.z_size = original_model.z_size

        self.name = original_model.name+'_dis'
        self.path = original_model.path

        self.model = np.zeros(self.original_model.shape)

    def gausswin(self, N, alpha, shift=0):
        """ 
        Gaussian window function with an optional shift.
        
        Parameters:
        - N: Length of the window.
        - alpha: Parameter of the Gaussian window.
        - shift: Shift of the Gaussian window.

        Returns:
        - w: Gaussian window.
        """
        n = np.arange(0, N) - (N - 1) / 2 - shift 
        window = np.exp(-(1/2) * (alpha * n / ((N - 1) / 2))**2)
        return window
    
    def exponential_field(self, X, Z, alpha, x0=None, z0=None):
        """ 
        Exponential field function. 
        
        Parameters:
        - X: x coordinate.
        - Z: z coordinate.
        - alpha: Parameter of the exponential field.
        - x0: x coordinate of the center of the exponential field.
        - z0: z coordinate of the center of the exponential field.
        
        Returns:
        - Exponential field.
        """
        if x0 is None:
            x0 = self.x_size / 2
        if z0 is None:
            z0 = self.z_size

        return np.exp(-alpha * np.sqrt((X - x0)**2 + (Z - z0)**2))

    def displace_inclusions_linear(self, max_disp_bottom, z_size):
        """
        Displace the inclusions linearly from a maximal displacement at the bottom to zero at the top.

        Parameters:
        - max_disp_bottom: Maximal displacement at the bottom of the domain.
        - z_size: The total vertical size of the domain.

        Returns:
        None
        """
        for idx, (x, z, radius) in enumerate(self.water_inclusion_pos):
            # Calculate the linear displacement factor for this inclusion
            # Linear factor scales from 0 (at top, z=0) to 1 (at bottom, z=z_size)
            linear_factor = z / z_size

            # Calculate displacement based on the linear factor and maximal displacement
            disp = linear_factor * max_disp_bottom

            # Update positions, assuming horizontal (x-axis) displacement
            new_x = x + disp  # or -disp if you want to displace in the opposite direction
            # z position remains unchanged for this simplistic model
            new_z = z

            # Update the inclusion position with the new values
            self.new_water_inclusion_pos[idx] = [new_x, new_z, radius]

    def displace_inclusions_gauss(self, lambda_val, alpha):
        """ 
        Displace the inclusions based on a repulsive Gaussian field.
        
        Parameters:
        - lambda_val: Displacement parameter.
        - alpha: Gaussian window parameter.

        Returns:
        None
        """
        # Define the grid
        nx = round(self.x_size / self.dx) 
        nz = round(self.z_size / self.dz)
        
        xpos = np.linspace(0, self.x_size, nx) 
        zpos = np.linspace(0, self.z_size, nz) 
        
        # Create the Gaussian window displacement for both axes
        new_xpos = xpos + lambda_val * self.gausswin(nx, alpha)
        zvec_func = lambda_val * self.gausswin(2 * nz, alpha)
        new_zpos = zpos + zvec_func[:nz] 
        
        # Compute change matrices
        xchange_mat = np.outer(new_xpos - xpos, new_zpos - zpos)
        zchange_mat = xchange_mat.copy()  # As per the MATLAB code provided

        xchange_mat = xchange_mat*0
        self.plot_displacement_matrix(xchange_mat, zchange_mat)

        for idx, (x, z, radius) in enumerate(self.water_inclusion_pos):
            # Find the nearest grid index
            ix, iz = int(x), int(z)  # Adjusted to consider grid spacing
            
            # Displacement proportional to gradient (i.e., slope)
            disp_x = xchange_mat[ix, iz]
            disp_z = zchange_mat[ix, iz]

            # Update positions
            new_x = x + disp_x
            new_z = z + disp_z

            # Update the new water inclusion position
            self.new_water_inclusion_pos[idx] = [new_x, new_z, radius]
        
        # Print the maximum displacement on the x-axis and z-axis
        print("Maximum displacement on the x-axis: ", round(self.dx * np.max(xchange_mat), 2))
        print("Maximum displacement on the z-axis: ", round(self.dz * np.max(zchange_mat), 2))

    def displace_inclusions_grad(self, lambda_val, alpha):
        """ 
        Displace the inclusions based on a repulsive Gaussian field. 
        
        Parameters:
        - lambda_val: Displacement parameter.
        - alpha: Gaussian window parameter.

        Returns:
        None
        """
        nx = round(self.x_size / self.dx)
        nz = round(self.z_size / self.dz) 
            
        # Creating the 2D Gaussian hill in the middle of the domain
        gaussian_hill = self.gausswin(nx, alpha)[:, np.newaxis] * self.gausswin(nz, alpha, shift=nz/2)

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
        """
        Apply the inclusions to the model.

        Parameters:
        None

        Returns:
        None
        """
        nx = round(self.x_size / self.dz)
        ny = round(self.y_size / self.dy)
        nz = round(self.z_size / self.dx)

        # Initialize the model
        self.model = np.zeros((nx, ny, nz), dtype=int)
        self.model[:, :, round(5.0 / self.dz):nz] = 1 # Glacier = 1
        
        # Create a grid for i and j
        i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')

        for inclusion in tqdm(self.new_water_inclusion_pos, desc="Processing inclusions"):
            # Calculate the squared distance for all points at once
            mask = ((j - inclusion[1]) ** 2 + (i - inclusion[0]) ** 2) <= ((inclusion[2] / self.dz) ** 2)
            
            # Reshape the mask
            mask_reshaped = mask[:, np.newaxis, :]

            # Use the reshaped mask to update the model
            self.model[mask_reshaped] = 3

        self.model[:, :, 0:round(5.0/self.dz)] = 0 # Freespace = 0
        self.model[:, :, round(105.0/self.dz):nz] = 2 # Bedrock = 2

    def displace(self):
        """
        Displace the inclusions in the model and store the result in self.model.
        
        Parameters:
        - lambda_val: Displacement parameter.
        - alpha: Gaussian window parameter.
        """
        lambda_val=3.6 
        alpha=3.5

        max_disp_bottom = .4 * 6

        # self.displace_inclusions_grad(lambda_val, alpha)
        self.displace_inclusions_linear(max_disp_bottom, self.z_size)
        self.apply_inclusions() 
        self.plot_displacement_scatter() 

    def plot_displacement_matrix(self, xchange_mat, zchange_mat):
        """
        Plot the displacement matrix.

        Parameters:
        None

        Returns:
        None
        """
        #Subplot the displacement matrix xchange_mat and zchange_mat
        plt.subplot(1, 2, 1)
        plt.imshow(xchange_mat.T*self.dx)
        plt.title("xchange_mat")
        plt.colorbar()
        plt.xlabel("nx")
        plt.ylabel("nz")
        plt.subplot(1, 2, 2)
        plt.imshow(zchange_mat.T*self.dz)
        plt.title("zchange_mat")
        plt.colorbar()
        plt.xlabel("nx")
        plt.ylabel("nz")
        plt.tight_layout()
        plt.savefig(self.path+'figures/'+self.name+'_matrices.png')
        plt.close()

    def plot_displacement_scatter(self):
        """
        Plot the displacement of the inclusions.

        Parameters:
        None

        Returns:
        None
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.water_inclusion_pos[:, 0]*self.dx, self.water_inclusion_pos[:, 1]*self.dy, 'o')
        plt.plot(self.new_water_inclusion_pos[:, 0]*self.dx, self.new_water_inclusion_pos[:, 1]*self.dy, 'x')
        plt.plot(self.x_size/2, self.z_size, 'v', markersize=10)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.xlim(0, self.x_size)
        plt.ylim(self.z_size, 0)
        plt.ylabel('depth [m]')
        plt.xlabel('distance [m]')
        plt.legend(['Original', 'Displaced', 'Source'])
        plt.title(self.name)
        plt.savefig(self.path+'figures/'+self.name+'scatter.png')
        plt.close()

        # save coordinates to csv files
        np.savetxt(self.path+'coordinates/'+self.name+'_original.csv', 
                   self.water_inclusion_pos, 
                   header='x,z,radius',
                   comments='',
                   delimiter=',')
        np.savetxt(self.path+'coordinates/'+self.name+'_displaced.csv', 
                   self.new_water_inclusion_pos, 
                   header='new_x,new_z,radius',
                   comments='',
                   delimiter=',')

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
        nx = round(self.x_size / self.dx)
        nz = round(self.z_size / self.dz)

        # Create an array for X and Y with appropriate shapes
        X, Y = np.meshgrid(np.linspace(0, self.x_size, nx), 
                        np.linspace(0, self.z_size, nz))
        
        plt.pcolormesh(X, Y, self.model[:, round(transceiver[1]*self.dy), :].T)
        plt.scatter(transceiver[0], transceiver[2])
        plt.scatter(receiver[0], receiver[2])
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.clim(0, 3)
        plt.ylabel('depth [m]')
        plt.xlabel('distance [m]')
        plt.title(self.name)
        plt.savefig(self.path+'/figures/'+self.name+'.png')
        plt.close()