from gprMax.gprMax import api
from tools.outputfiles_merge  import merge_files

class SimulationRunner:
    """
    Class to run simulations
    """

    def __init__(self, simulation_model):
        """
        init SimulationRunner

        Parameters:
        self (SimulationRunner): the SimulationRunner to initialize
        simulation_model (SimulationModel): the simulation model to run

        Returns:
        None
        """
        self.simulation_model = simulation_model
        print(simulation_model.path)
        print(simulation_model.name)
    def run_simulation(self, measurement_number, gpu_number, gpu_set):
        """
        Run the gprMax simulation using gprMax API. GPU number and threads are hardcoded.

        Parameters:
        self (SimulationRunner): the SimulationRunner to run
        measurement_number (int): the number of measurements

        Returns:
        None
        """

        api(self.simulation_model.path + self.simulation_model.name + '.in', mpi = gpu_number, gpu = gpu_set, n = measurement_number)
    
    def merge_files(self, remove_files):
        """
        Merge files using gprMax included function

        Parameters:
        self (SimulationRunner): the SimulationRunner to merge files
        remove_files (bool): whether to remove the files after merging

        Returns:
        None
        """
        merge_files(self.simulation_model.path + self.simulation_model.name, removefiles = remove_files)