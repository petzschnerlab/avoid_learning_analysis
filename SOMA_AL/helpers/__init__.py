from helpers.SOMA_processing import SOMAProcessing
from helpers.SOMA_plotting import SOMAPlotting
from helpers.SOMA_report import SOMAReport
from helpers.SOMA_parameters import SOMAParameters
from helpers.SOMA_tests import SOMATests
from helpers.SOMA_statistics import SOMAStatistics

class SOMAMaster(SOMAProcessing, 
                 SOMAPlotting, 
                 SOMAReport, 
                 SOMAParameters,
                 SOMAStatistics, 
                 SOMATests):
    
    """
    Class to hold all functions for the SOMA project
    """

    def __init__(self):
        super().__init__()





