import numpy as np
import matplotlib.pyplot as plt

class SimulationModel:
    # This class is used to store the model information and generate the base model.

    def __init__(self, name, x_size, y_size, z_size, 
                 discrete, materials, path):
        self.name = name
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.discrete = discrete
        self.materials = materials
        self.path = path

    def calculate_measurment_step(self, number_of_measurements, antenna_spacing):
        nx = self.model.shape[0]
        nx_buffered = nx - 50 # buffer of 20 cell on each side
        return round((nx_buffered * self.discrete[0] - antenna_spacing) / number_of_measurements, 2)

    def generate_base_glacier(self):
        nx = int(self.x_size / self.discrete[0])
        ny = int(self.y_size / self.discrete[1])
        nz = int(self.z_size / self.discrete[2])
        
        self.model = np.zeros((nx, ny, nz)) # Free space = 0
        self.model[:, :, round(5.0 / self.discrete[2]):nz] = 1 # Glacier = 1

    import numpy as np

    def water_inclusion(self, LWC = 0.1, s_max = 0.2, hb = 100, f = 1):
        """
        Include water inclusions in the model.

        Parameters:
        LWC: liquid water content in %
        s_max: maximum radius of inclusion (in m). Remark: no minimal size
        hb: bedrock thickness (we don't want water inclusions where the bedrock is)
        f: fraction of water saturated layer, i.e. the ice layer that contains the water inclusions (starting from the bedrock)

        Returns:
        m: a logical array with 1 for water presence
        nb: number of scatterers
        """

        lx = self.x_size
        ly = self.y_size
        h = self.discrete[0] # assuming y discretization
        X = int(lx / h)  # image size along x axis, in pixel
        Y = int(ly / h)  # image size along y axis, in pixel

        # Scatter inclusions
        col, rows = np.meshgrid(np.arange(1, X+1), np.arange(1, Y+1))

        m = np.zeros((Y, X), dtype=bool)  # m is a 2D "logical" matrix of the "dry" water layer (without inclusion).

        liq = 0  # initiation of liquid water content
        nb = 0   # initial counts of scatterers

        while liq < LWC:
            x = np.random.rand() * X  # center location in pixel
            y = np.random.rand() * Y  # note that repeatability is allowed, but shouldn't be an issue for small LWC values as it is in our case (~1%)
            r = np.random.rand() * s_max  # in meters
            
            # we only create water inclusion above bedrock and below the water saturated layer defined by f
            if y < Y - hb/h and y > Y * (1 - f):
                m = np.logical_or(m, (rows - y)**2 + (col - x)**2 <= (r/h)**2)  # in pixel
                liq += (np.pi * r**2 / (lx * ly)) * 100  # LWC in 2D in % (m2)
                nb += 1

        nb -= 1  # nb was incremented by one before exiting the while loop
        return m, nb

    def plot_initial_model(self, transceiver, receiver):

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
        plt.savefig(self.path+'/figures/'+self.name+'.png')
        plt.close()        