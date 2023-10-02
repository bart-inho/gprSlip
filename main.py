# This program is used to run the gprMax simulation applied to glaciology.
from services.file_service import FileService
from services.folder_init import InitializeFolders
from models.simulation_model import SimulationModel
from models.material import Material
from simulations.simulation_runner import SimulationRunner
from simulations.simulation_plot_profile import PlotProfile  
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='run the simulation')
    parser.add_argument('--plot', action='store_true', help='plot the simulation')
    parser.add_argument('--rough', action='store_true', help='rough bedrock')
    args = parser.parse_args()

    # Initialize folders
    InitializeFolders.check_and_create_directories()

    # Initialize Materials
    # Change the material names in the "Material" class 
    freespace = Material(1. , 0.   , 1., 0., 'freespace') # Free space
    glacier   = Material(3.2, 5.e-8, 1., 0., 'glacier'  ) # Glacier
    bedrock   = Material(5. , 1.e-2, 1., 0., 'bedrock'  ) # Bedrock
    water     = Material(80. , 1.e-10   , 1., 0., 'water'    ) # Water
    
    # Initialize SimulationModel
    model_name    = 'test_temperate_glacier'
    inout_files   = 'inout_files/'
    path_to_files = inout_files + model_name

    dis = 0.08

    # Generate model
    model = SimulationModel(model_name, 
                            100+dis, dis, 110+dis, 
                            [dis, dis, dis], # Change discretisation if needed here
                            [freespace, glacier, bedrock, water], # Change name of materials here
                            inout_files)
    
    # Generate base model
    model.generate_base_glacier()
    model.water_inclusion()

    measurement_number = 100 # number of traces
    antenna_spacing    = 4  # Change antenna spacing in [m] here
    measurement_step   = model.calculate_measurment_step(measurement_number, 
                                                         antenna_spacing) # Change antenna spacing in m here
     
    # Add antenna positions
    transceiver1 = [round(25 * model.discrete[0]), # 25 cells of buffer (20 minimum)    
                    0,
                    4.5] # 0.5 cm above the glacier surface
    
    receiver1    = [round(25 * model.discrete[0] + antenna_spacing), # 25 cells of buffer (20 minimum)
                    0,
                    4.5] # 0.5 cm above the glacier surface
        
    #Plot initial model
    model.plot_initial_model(transceiver1, receiver1)

    # Call FileService to write files
    FileService.write_materials_file(model.path + model.name + '_materials', 
                                     model.materials)
    
    FileService.write_h5_file(model.path + model.name + '_h5', 
                              model)

    FileService.write_input_file(model, 
                                path_to_files, 
                                path_to_files + '_materials', 
                                path_to_files + '_h5', 
                                25e6,   # Change frequency in Hz here
                                transceiver1, receiver1, 
                                measurement_step, 
                                1000e-9) # Change time window in s here
        
    # Run simulation
    if args.run:
        simulation_runner = SimulationRunner(model)
        simulation_runner.run_simulation(measurement_number)
        simulation_runner.merge_files(True)
        
    # Plot profile
    if args.plot:
        plot_profile = PlotProfile(model.path + model.name + '_merged.out', 'Ey')
        plot_profile.get_output_data()
        plot_profile.plot()

if __name__ == "__main__":
    main()