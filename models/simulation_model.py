import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SimulationModel:
    """
    Class to represent the simulation model
    """

    def __init__(self, name, x_size, y_size, z_size, 
                 discrete, materials, path):
        """
        Initialize the model

        Parameters:
        self (SimulationModel): the model to initialize
        name (str): the name of the model
        x_size (float): the x size of the model
        y_size (float): the y size of the model
        z_size (float): the z size of the model
        discrete (list): the discretization of the model
        materials (list): the materials of the model
        path (str): the path to the model

        Returns:
        None
        """
        self.name = name
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.discrete = discrete
        self.materials = materials
        self.path = path

    def calculate_measurment_step(self, number_of_measurements, antenna_spacing):
        """ 
        Calculate the measurement step 

        Parameters:
        self (SimulationModel): the model to calculate the measurement step from
        number_of_measurements (int): the number of measurements
        antenna_spacing (int): the antenna spacing

        Returns:
        float: the measurement step
        """

        right_buffer = 5. # buffer of 5 meters on the right side
        left_buffer = 5. # buffer of 5 meters on the left side

        nx_buffered = self.model.shape[0] - (right_buffer + left_buffer + antenna_spacing)/self.discrete[0]
        return round(nx_buffered * self.discrete[0] / number_of_measurements, 2)

    def generate_base_glacier(self, h_freespace=5., h_bedrock=105.):
        """
        Generate base model

        Parameters:
        self (SimulationModel): the model to generate

        Returns:
        none
        """

        nz = int(self.z_size / self.discrete[2])
        self.model[:, :, 0:round(h_freespace/self.discrete[2])] = 0 # Freespace = 0
        self.model[:, :, round(h_bedrock/self.discrete[2]):nz] = 2 # Bedrock = 2

    def include_inclusions_loop(self, nx, nz, max_inclusion_radius, num_inclusions=5000):
        water_matrix = np.zeros((nz, nx), dtype=bool)
        cols, rows = np.meshgrid(np.arange(nx), np.arange(nz))
        water_inclusion_pos = []
        total_liquid_content = 0


        for _ in tqdm(range(num_inclusions), desc='Creating water inclusions'):

            x_center_pixel = np.random.rand() * nx
            z_center_pixel = np.random.rand() * nz
            radius = max_inclusion_radius

            water_inclusion_pos.append([x_center_pixel, z_center_pixel, radius])
            water_matrix = np.logical_or(water_matrix, ((rows - z_center_pixel) ** 2 + (cols - x_center_pixel) ** 2 <= (radius / self.discrete[2]) ** 2))

            new_liquid_content = (np.pi * radius ** 2 / (self.x_size * self.z_size)) * 100
            total_liquid_content += new_liquid_content
        
        return water_matrix, water_inclusion_pos, total_liquid_content

    def include_inclusions_wc(self, nx, nz, max_inclusion_radius, liquid_water_content):
        water_matrix = np.zeros((nz, nx), dtype=bool)
        cols, rows = np.meshgrid(np.arange(nx), np.arange(nz))
        water_inclusion_pos = []
        total_liquid_content = 0

        with tqdm(total=liquid_water_content) as pbar:
            while total_liquid_content < liquid_water_content:
                x_center_pixel = np.random.rand() * nx
                z_center_pixel = np.random.rand() * nz
                radius = max_inclusion_radius

                water_inclusion_pos.append([x_center_pixel, z_center_pixel, radius])
                water_matrix = np.logical_or(water_matrix, ((rows - z_center_pixel) ** 2 + (cols - x_center_pixel) ** 2 <= (radius / self.discrete[2]) ** 2))

                new_liquid_content = (np.pi * radius ** 2 / (self.x_size * self.z_size)) * 100
                total_liquid_content += new_liquid_content
                pbar.update(round(new_liquid_content, 5))
        
        return water_matrix, water_inclusion_pos, total_liquid_content

    def water_inclusion(self, 
                        liquid_water_content=0.1, 
                        number_of_inclusions=5000,
                        max_inclusion_radius=0.05,
                        h_glacier=5.):
        """
        Adds water inclusions to the glacier model.
        
        Parameters:
        liquid_water_content (float): liquid water content in %.
        max_inclusion_radius (float): maximum radius of water inclusion in meters.
        """

        water_inclusion_pos = [] # Initialize a list to store the water inclusion positions
        i = 0 # Initialize a counter

        nx = round(self.x_size / self.discrete[0])
        ny = round(self.y_size / self.discrete[1])
        nz = round(self.z_size / self.discrete[2])

        # Initialize the model
        self.model = np.zeros((nx, ny, nz), dtype=int)
        self.model[:, :, round(h_glacier / self.discrete[2]):nz] = 1 # Glacier = 1

        # Initiate a matrix to represent water presence (1 for water presence)
        water_matrix = np.zeros((nz, nx), dtype=bool)
        
        total_liquid_content = 0 # initialize the total liquid content
        
        # Add water inclusions until the total liquid content is reached
        # water_matrix, total_liquid_content = self.include_inclusions_wc(nx, nz, max_inclusion_radius, liquid_water_content)
        water_matrix, water_inclusion_pos, total_liquid_content = self.include_inclusions_loop(nx, nz, max_inclusion_radius, number_of_inclusions)
        
        print('Total liquid content: ', round(total_liquid_content, 3), '%')

        # Update the model to include the water inclusions
        for i in range(nz):
            self.model[water_matrix[i], :, i] = 3  # Assuming that water is represented by the value 3 in the model

        return np.array(water_inclusion_pos)

    def plot_initial_model(self, transceiver, receiver):
        """
        Plot the initial model

        Parameters:
        self (SimulationModel): the model to plot
        transceiver (np.array): the transceiver position
        receiver (np.array): the receiver position

        Returns:
        None
        """

        nx = round(self.x_size / self.discrete[0])
        nz = round(self.z_size / self.discrete[2])

        # Create an array for X and Y with appropriate shapes
        X, Y = np.meshgrid(np.linspace(0, self.x_size, nx), 
                        np.linspace(0, self.z_size, nz))
        
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
        plt.savefig(self.path+'/figures/'+self.name+'.png')
        plt.close()        