# GprMax Simulation for Temperate Glaciers

This script uses gprMax, a finite-difference time-domain electromagnetic simulation software, to simulate ground penetrating radar scenarios applied to glaciology. It provides the ability to set up simulations, plot the initial state of the simulation, and visualize the results.

## Requirements:
User must have gprMax installed and properly set up on their system. For more information, refer to the [gprMax documentation](https://docs.gprmax.com/en/latest/). gprSlip folder should be placed in the main gprMax folder.

## Features:

1. Initialize and visualize the geometry of a glacier model.
2. Incorporate material properties of freespace, glacier, bedrock, and water.
3. Generate random water inclusions in the glacier to simulate temperate glacier conditions.
4. Plot the initialized glacier model with antenna positions.
5. Output required input, material, and geometry files for gprMax.
6. Execute gprMax simulations and visualize the results.

## Usage:

### Command-Line Options:

- `--run`: Run the simulation.
- `--plot`: Plot the simulation results.
- `--rough`: Incorporate rough bedrock (this feature is mentioned but not yet implemented in the provided code).

Example usage:

```
python your_script_name.py --run --plot
```

### Model Configuration:

- Modify material properties in the `Material` class as required.
- Change the glacier model dimensions and discretizations in the `SimulationModel` initialization.
- Adjust the antenna positions, spacing, and measurement settings as needed.

## Files and Classes:

### Main Classes:

1. `SimulationModel`: Represents the glacier model, its geometry, and other attributes. Contains methods to generate the glacier base and water inclusions, and to plot the initial state.
2. `Material`: Represents materials (e.g., glacier, bedrock) and their electromagnetic properties.
3. `FileService`: Contains methods to write the required input, materials, and geometry files for gprMax.

### Dependencies:

- `numpy`: Used for numerical operations and array manipulations.
- `matplotlib`: Used for plotting the initial state of the glacier model.
- `h5py`: Used for writing the model's geometry to an HDF5 file, which is required for gprMax.
- Various service modules (not provided in the script): `file_service`, `folder_init`, `simulation_runner`, `simulation_plot_profile`.

## 3. Running Simulations with `SimulationRunner`

Once you've defined a `SimulationModel`, you can easily run a simulation using the `SimulationRunner` class. This class takes care of the actual simulation execution using the gprMax API.

### Example:

```python
from gprMax.gprMax import SimulationModel
from your_script_name import SimulationRunner

# Assuming you've defined a SimulationModel
sim_model = SimulationModel(path="/path/to/your/model/", name="model_name")

runner = SimulationRunner(simulation_model=sim_model)
runner.run_simulation(measurement_number=10)  # Run the simulation for 10 measurements
```

If you need to merge the simulation files, you can use the `merge_files` method:

```python
runner.merge_files(remove_files=True)  # Merge and remove the original files
```

## 4. Plotting the Simulation Results with `PlotProfile`

After running your simulations, you might want to visualize the results. Use the `PlotProfile` class for this.

### Example:

```python
from your_script_name import PlotProfile

# Create a PlotProfile instance
profile = PlotProfile(outputfile="/path/to/your/output/file", rx_component="Ez")  # Let's say you want to plot the Ez component

# Get the output data
profile.get_output_data()

# Plot the data
profile.plot()
```

This will automatically create a plot of your simulation results using `matplotlib`.

## Notes:
1. Ensure you have all required modules installed.
2. The plotting functions use `matplotlib` and `h5py` for reading the output data and visualizing it. Ensure you have both libraries installed.
3. For the plotting to work correctly, you should provide the correct `rx_component` which denotes which component you want to visualize (e.g., 'Ez', 'Ex', 'Hy', etc.).
4. The paths in the examples are placeholders. Make sure you replace them with the appropriate paths specific to your setup.


## Contribution:

Feel free to raise issues or submit pull requests for additional features or improvements.