import os
import h5py
class FileService:
    """
    Class to write input files for gprMax simulations 
    """

    @staticmethod
    def write_materials_file(path_to_materials, materials):
        """
        Write the materials file

        Parameters:
        path_to_materials (str): the path to the materials file
        materials (list): the materials to write

        Returns:
        None        
        """
        with open(path_to_materials+'.txt', 'w') as file: 
            for material in materials:
                file.write('#material: {} {} {} {} {}\n'.format(material.eps_r, 
                                                                material.sigma, 
                                                                material.mu_r, 
                                                                material.vel, 
                                                                material.name))

    @staticmethod
    def write_input_file(model, path_to_input, path_to_materials, path_to_h5, 
                         freq, transiever, reciever, mstep, time_window):
        """
        Write the input file

        Parameters:
        model (SimulationModel): the model to write the input file for
        path_to_input (str): the path to the input file
        path_to_materials (str): the path to the materials file
        path_to_h5 (str): the path to the h5 file
        freq (float): the frequency of the wave
        transiever (list): the transiever position
        reciever (list): the reciever position
        mstep (float): the measurement step
        time_window (float): the time window

        Returns:
        None
        """
        with open(path_to_input+'.in', 'w') as file:
            dx, dy, dz = model.discrete
            file.write('#title: {}\n'.format(model.name))
            file.write('#domain: {} {} {}\n'.format(model.x_size, model.y_size, model.z_size))
            file.write('#dx_dy_dz: {} {} {}\n'.format(dx, dy, dz))
            file.write('#time_window: {}\n'.format(time_window))
            file.write('#waveform: ricker 1 {} my_ricker\n'.format(freq))
            file.write('#hertzian_dipole: y {} {} {} my_ricker\n'.format(*transiever))
            file.write('#rx: {} {} {}\n'.format(*reciever))
            file.write('#src_steps: {} 0 0\n'.format(mstep))
            file.write('#rx_steps: {} 0 0\n'.format(mstep))
            file.write('#geometry_objects_read: 0 0 0 {} {}\n'.format(path_to_h5+'.h5', path_to_materials+'.txt'))

    @staticmethod
    def write_h5_file(path_to_h5, model):
        """
        Write the h5 file
        
        Parameters:
        path_to_h5 (str): the path to the h5 file
        model (SimulationModel): the model to write the h5 file for

        Returns:
        None
        """

        if os.path.exists(path_to_h5):
            os.remove(path_to_h5)
        with h5py.File(path_to_h5+'.h5', 'w') as hdf:
            modelh5 = hdf.create_dataset(name='data', data=model.model)
            hdf.attrs['dx_dy_dz'] = model.discrete