from helpers.processing import Processing
from helpers.plotting import Plotting
from helpers.report import Report
from helpers.parameters import Parameters
from helpers.tests import Tests
from helpers.statistics import Statistics

class Master(Processing, 
                 Plotting, 
                 Report, 
                 Parameters,
                 Statistics, 
                 Tests):
    
    """
    Class to hold all functions for the SOMA project
    """

    def __init__(self):
        super().__init__()





