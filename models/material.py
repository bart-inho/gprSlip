class Material:
    """
    Class to represent materials in the model
    """

    def __init__(self, eps_r, sigma, mu_r, vel, name):
        """
        Initialize the material

        Parameters:
        self (Material): the material to initialize
        eps_r (float): the relative permittivity of the material
        sigma (float): the conductivity of the material
        mu_r (float): the relative permeability of the material
        vel (float): the velocity of the material
        name (str): the name of the material

        Returns:
        None
        """
        self.eps_r = eps_r
        self.sigma = sigma
        self.mu_r = mu_r
        self.vel = vel
        self.name = name

    def __str__(self):
        """
        Return the material information

        Parameters:
        self (Material): the material to print

        Returns:
        str: the material information
        """
        return f"{self.eps_r} {self.sigma} {self.mu_r} {self.vel} {self.name}"