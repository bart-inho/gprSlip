# This program is used to run the gprMax simulation applied to glaciology.
from services.file_service import FileService
from services.folder_init import InitializeFolders
from models.simulation_model import SimulationModel
from models.inclusion_displacer import InclusionDisplacer
from models.material import Material
from simulations.simulation_runner import SimulationRunner
from simulations.simulation_plot_profile import PlotProfile  
import argparse

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store_true', help='generate the model')
    parser.add_argument('--run', action='store_true', help='run the simulation')
    parser.add_argument('--plot', action='store_true', help='plot the simulation')
    args = parser.parse_args()

    # Initialize folders
    InitializeFolders.check_and_create_directories()

    # Initialize folders paths
    model_name    = 'temp_ice'
    model_name_displaced = 'temp_ice_dis'
    inout_files   = 'inout_files/'
    path_to_files = inout_files + model_name

    # Initialize Materials
    freespace = Material(1. , 0.   , 1., 0., 'freespace') # Free space
    glacier   = Material(3.2, 5.e-8, 1., 0., 'glacier'  ) # Glacier
    bedrock   = Material(6. , 1.e-3, 1., 0., 'bedrock'  ) # Bedrock
    water     = Material(80. , 5.e-4   , 1., 0., 'water') # Water

    # SET SIMULATION PARAMETERS
    dis = 0.05 # Discretisation in m
    time_window = 1.5e-6 # Time window in s
    measurement_number = 200 # number of traces
    antenna_spacing    = 4  # Change antenna spacing in [m] here

    water_liquid_content = 0.1 # Water liquid content in %
    number_of_inclusions = 5000 # Number of inclusions
    max_radius_inclusions = 0.05 # Maximum radius of inclusions in m

    lambda_val = 3.6 # width of the gaussian pulse in m
    alpha = 3.5 # Attenuation in dB/m
    # Generate model
    model = SimulationModel(model_name, 
                            100+dis, dis, 110+dis, 
                            [dis, dis, dis], # Change discretisation if needed here
                            [freespace, glacier, bedrock, water], # Change name of materials here
                            inout_files)
    
    # Generate base model
    water_inclusion_pos = model.water_inclusion(water_liquid_content, 
                                                number_of_inclusions, 
                                                max_radius_inclusions)
    model.generate_base_glacier()

    # Displace inclusions
    model_dis = InclusionDisplacer(model, water_inclusion_pos)
    model_dis.displace()

    measurement_step   = model.calculate_measurment_step(measurement_number, 
                                                        antenna_spacing) # Change antenna spacing in m here
    
    # Add antenna positions
    transceiver1 = [5, # 25 cells of buffer (20 minimum)    
                    0, # It is a 2D model, so y = 0
                    4.5] # 0.5 cm above the glacier surface
    
    receiver1    = [5 + antenna_spacing, # 25 cells of buffer (20 minimum)
                    0, # It is a 2D model, so y = 0
                    4.5] # 0.5 cm above the glacier surface
    
    #Plot initial model
    print("Producing plots...")
    model.plot_initial_model(transceiver1, receiver1)
    model_dis.plot_displaced_model(transceiver1, receiver1)

    # Call FileService to write files
    FileService.write_materials_file(model.path + model.name + '_materials', 
                                    model.materials)
    
    FileService.write_h5_file(model.path + model.name + '_h5', model)

    FileService.write_input_file(model, 
                                path_to_files, 
                                path_to_files + '_materials', 
                                path_to_files + '_h5', 
                                25e6,   # Change frequency in Hz here
                                transceiver1, receiver1, 
                                measurement_step, 
                                time_window) # Change time window in s here
    
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

    # Adjust the file names for the displaced model
    path_to_files_displaced = inout_files + model_name_displaced

    # Write the files for the displaced model
    FileService.write_materials_file(model_dis.path + model_name_displaced + '_materials', model.materials)
    
    FileService.write_h5_file(model_dis.path + model_name_displaced + '_h5', model_dis)

    FileService.write_input_file(model_dis, 
                                path_to_files_displaced, 
                                path_to_files_displaced + '_materials', 
                                path_to_files_displaced + '_h5', 
                                25e6,   # Change frequency in Hz here
                                transceiver1, receiver1, 
                                measurement_step, 
                                time_window) # Change time window in s here

    # Run the displaced model simulation
    if args.run:
        simulation_runner_displaced = SimulationRunner(model_dis)
        simulation_runner_displaced.run_simulation(measurement_number)
        simulation_runner_displaced.merge_files(True)

    # Plot the displaced model
    if args.plot:
        plot_profile_displaced = PlotProfile(model_dis.path + model_name_displaced + '_merged.out', 'Ey')
        plot_profile_displaced.get_output_data()
        plot_profile_displaced.plot()

    print('Done!')

if __name__ == "__main__":
    main()