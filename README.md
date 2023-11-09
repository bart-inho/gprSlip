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

- `--model`: Generate the model files.
- `--run`: Run the simulation.
- `--plot`: Plot the simulation results.

Example usage:

```
python main.py --run --plot
```

### Model Configuration:

- Modify material properties in the `Material` class as required.
- Change the glacier model dimensions and discretizations in the `SimulationModel` initialization.
- Adjust the antenna positions, spacing, and measurement settings as needed.

## Files and Classes:

### Main Classes:

1. `SimulationModel`: Represents the glacier model, its geometry, and other attributes. Contains methods to generate the glacier base and water inclusions, and to plot the initial state.
2. `InclusionDisplacer`: Creates a second model that contains displaced inclusions for ice-quakes modelling.
3. `Material`: Represents materials (e.g., glacier, bedrock) and their electromagnetic properties.
4. `FileService`: Contains methods to write the required input, materials, and geometry files for gprMax.
5. `SimulationRunner`: Runs the simulation once the initial models are created
6. `PlotProfile`: Plots the generated radagrams for first insights.

### Dependencies:

- `numpy`: Used for numerical operations and array manipulations.
- `matplotlib`: Used for plotting the initial state of the glacier model.
- `h5py`: Used for writing the model's geometry to an HDF5 file, which is required for gprMax.
- Various service modules (not provided in the script): `file_service`, `folder_init`, `simulation_runner`, `simulation_plot_profile`.

## 3. Running Simulations with `SimulationRunner`

Once you've defined a `SimulationModel`, you can easily run a simulation using the `SimulationRunner` class. This class takes care of the actual simulation execution using the gprMax API.

## Notes:
1. Ensure you have all required modules installed.
2. The plotting functions use `matplotlib` and `h5py` for reading the output data and visualizing it. Ensure you have both libraries installed.
3. For the plotting to work correctly, you should provide the correct `rx_component` which denotes which component you want to visualize (e.g., 'Ez', 'Ex', 'Hy', etc.).
4. The paths in the examples are placeholders. Make sure you replace them with the appropriate paths specific to your setup.


## Contribution:

Feel free to raise issues or submit pull requests for additional features or improvements.
