# gprSlip

## Overview
`gprSlip` is an advanced simulation toolkit for modeling the Ground Penetrating Radar (GPR) response within temperate glaciers. It utilizes the `gprMax` software to account for the complexities of water inclusions in glacial structures. This tool allows for the creation of detailed glacial models, the simulation of GPR signal propagation, and the visualization of the simulation outputs.

## Features
- Customizable 2D glacial model generation with a variety of material parameters.
- Dynamic inclusion of water bodies with tunable volume and distribution.
- GPR signal simulation in glacial environments with high fidelity.
- Support for multi-GPU computation to enhance simulation performance.
- Plotting capabilities for both initial and modified glacial models.

## Prerequisites
`gprSlip` requires a working installation of `gprMax`, which is a highly flexible and widely used tool for simulating GPR. Ensure that `gprMax` is installed as per the official documentation found at [gprMax Documentation](https://www.gprmax.com/documentation).

## Installation
To set up `gprSlip`, follow these steps after installing `gprMax`:

1. Clone the `gprSlip` repository into the main directory of your `gprMax` installation:

```sh
cd path/to/gprMax
git clone https://github.com/bart-inho/gprSlip.git
```

2. Navigate to the `gprSlip` directory:

```sh
cd gprSlip
```

## Configuration
Before running a simulation, you may want to adjust the model parameters within `main.py`. These include:

- GPRMax settings: `dis`, `time_window`, `measurement_number`, `antenna_spacing`, `frequency`
- Geometric properties: `glacier_length`, `glacier_thickness`, `buffer_antenna`, `h_antenna`
- Water inclusion characteristics: `water_liquid_content`, `number_of_inclusions`, `max_radius_inclusions`
- Simulation parameters: `gpu_number`, `gpu_set`, `merge_file`

## Usage
With `gprSlip`, you can generate models, run simulations, and plot results using the following command-line options:

```sh
python main.py [--model] [--run] [--plot]
```

- `--model`: Constructs the glacial model based on specified parameters.
- `--run`: Initiates the simulation process using the generated model.
- `--plot`: Produces graphical plots to visualize the simulation data.

## Output
`gprSlip` will generate several output files that represent the simulated GPR data. These can be reviewed directly or used in conjunction with other data analysis tools for further investigation.

## Contributing
We welcome contributions to the `gprSlip` project. To contribute:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Create a new Pull Request.