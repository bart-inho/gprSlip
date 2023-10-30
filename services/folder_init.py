from pathlib import Path

class InitializeFolders:
    """
    Class to initialize folders
    """

    @staticmethod
    def check_and_create_directories():
        """
        Check if the directories exist, if not create them
        
        Parameters:
        None

        Returns:
        None
        """
        Path('inout_files/figures').mkdir(parents=True, exist_ok=True)
        Path('inout_files/coordinates').mkdir(parents=True, exist_ok=True)