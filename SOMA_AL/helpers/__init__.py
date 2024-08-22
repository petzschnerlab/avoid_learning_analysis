from helpers.SOMA_processing import SOMAProcessing
from helpers.SOMA_plotting import SOMAPlotting
from helpers.SOMA_report import SOMAReport
from SOMA_AL.helpers.SOMA_parameters import SOMAParameters
from helpers.SOMA_tests import SOMATests

class SOMAMaster(SOMAProcessing, 
                 SOMAPlotting, 
                 SOMAReport, 
                 SOMAParameters, 
                 SOMATests):
    
    """
    Class to hold all functions for the SOMA project
    """

    def __init__(self):
        super().__init__()





