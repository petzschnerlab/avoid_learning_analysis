#Import modules
import warnings
import os

class Parameters:

    """
    Class to set parameters for the SOMA pipeline
    """

    #Project organization
    def set_parameters(self, **kwargs: dict) -> None:
        
        """
        Assigns parameters to the SOMA pipeline class

        Parameters
        ----------
        kwargs : dict
            The parameters to assign to the SOMA pipeline class, see help for details.

        Returns (Internal)
        ------------------
        self.figure_count : int
            The figure count for the report.
        self.table_count : int
            The table count for the report.
        self.repo_directory : str
            The directory of the AL repository.
        self.kwargs : dict
            The parameters to assign to the SOMA pipeline class.
        self.help : bool
            Whether to print the help information.
        self.author : str
            The author of the report.
        self.rscripts_path : str
            The path to the R scripts.
        self.file_path : str
            The path to the data file.
        self.file_name : str
            The name of the data file.
        self.dataset : str
            The name of the dataset.
        self.print_filename : str
            The filename for the report.
        self.split_by_group : str
            The group to split the data by.
        self.split_by_group_id : str
            The group ID to split the data by.
        self.pain_cutoff : int
            The cutoff for the pain group.
        self.depression_cutoff : int
            The cutoff for the depression group.
        self.accuracy_exclusion_threshold : int
            The threshold for excluding participants based on accuracy.
        self.RT_low_threshold : int
            The lower threshold for excluding trials based on reaction times.
        self.RT_high_threshold : int
            The upper threshold for excluding trials based on reaction times.
        self.rolling_mean : int
            The number of trials to use in the rolling mean for plotting learning curves.
        self.tests : str
            The type of tests to run.
        self.test_rolling_mean : int
            The number of trials to use in the rolling mean for the tests learning curves.
        self.test_context_type : str
            The context type to use in the tests learning curves.
        self.hide_stats : bool
            Whether to hide the statistics in the report.
        self.load_stats : bool
            Whether to load the statistics from the backend.
        self.hide_posthocs : bool
            Whether to hide the posthocs in the report.
        self.load_posthocs : bool
            Whether to load the posthocs from the backend.
        self.load_models : bool
            Whether to load the models from the backend.
        self.verbose : bool
            Whether to print verbose output.
        """

        #Warning of unknown params
        accepted_params = ['help',
                           'author',
                           'rscripts_path',
                           'file_path', 
                           'file_name', 
                           'print_filename', 
                           'split_by_group',
                           'split_by_group_id',
                           'dataset',
                           'pain_cutoff',
                           'depression_cutoff',
                           'accuracy_exclusion_threshold',
                           'RT_low_threshold',
                           'RT_high_threshold',
                           'rolling_mean',
                           'tests',
                           'test_rolling_mean',
                           'test_context_type',
                           'hide_stats',
                           'load_stats',
                           'hide_posthocs',
                           'load_posthocs',
                           'load_models',
                           'verbose']
        
        #Check for unknown parameters
        for key in kwargs:
            if key not in accepted_params:
                warnings.warn(f'Unknown parameter {key} is being ignored', stacklevel=2)

        #Warning of missing required params
        required_params = ['file_path', 
                           'file_name']
        
        if not kwargs.get('help', False):
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f'Missing required parameter {param}, which does not contain a default. Please provide a value for this parameter.')

        #Set internal parameters
        self.figure_count = 1
        self.table_count = 1
        self.repo_directory = os.path.dirname(os.path.realpath(__file__)).split('AL')[0]

        #Unpack parameters
        self.kwargs = kwargs
        defaults = {
            'help': False,
            'author': 'PEAC_Team',
            'rscripts_path': None,
            'file_path': None,
            'file_name': None,
            'dataset': '',
            'print_filename': r'AL/reports/PEAC_report.pdf',
            'split_by_group': 'pain',
            'split_by_group_id': kwargs.get('split_by_group', 'pain'),
            'pain_cutoff': 2,
            'depression_cutoff': 10,
            'accuracy_exclusion_threshold': 70,
            'RT_low_threshold': 200,
            'RT_high_threshold': 5000,
            'rolling_mean': 5,
            'tests': 'basic',
            'test_rolling_mean': None,
            'test_context_type': 'context',
            'hide_stats': False,
            'load_stats': False,
            'hide_posthocs': False,
            'load_posthocs': False,
            'load_models': False,
            'verbose': False,
        }
        for key, default in defaults.items():
            setattr(self, key, kwargs.get(key, default))

        #Format parameters
        self.split_by_group = self.split_by_group if ',' not in self.split_by_group else self.split_by_group
        self.hide_posthocs = True if self.hide_stats == True else self.hide_posthocs
        self.print_filename = self.print_filename.replace('.pdf', f'_{self.split_by_group_id}.pdf')
        self.group_code = 'group_code' if self.split_by_group == 'pain' else 'depression'
        if self.split_by_group == 'pain':
            self.group_labels = ['no pain', 'acute pain', 'chronic pain']
            self.group_labels_formatted = ['No\nPain', 'Acute\nPain', 'Chronic\nPain']
        else:
            self.group_labels = ['healthy', 'depressed']
            self.group_labels_formatted = ['Healthy', 'Depressed']

        #Create needed directories
        if not os.path.exists(f"AL/plots/{self.split_by_group}"):
            os.makedirs(f"AL/plots/{self.split_by_group}")

        if not os.path.exists(f"AL/modelling/"):
            os.makedirs(f"AL/modelling/")

    #Print chosen parameters
    def announce(self, case: str = 'start') -> None:

        """
        Prints the parameters chosen for the SOMA pipeline

        Parameters
        ----------
        case : str
            The case to announce the parameters, either 'start' or 'end'. The default is 'start'.
        """

        if case == 'start' and self.verbose:
            print('\n*******************************************')
            if self.author != 'SOMA_Team':
                print(f'Welcome {self.author.replace("_"," ").title()}!')
            print(f'\nRunning the SOMA pipeline with the following parameters:\n')
            [print(f'{key}: {value}') for key, value in self.kwargs.items()]
            print(f'\nProcessing {self.split_by_group} group data...')
            print('*******************************************\n')
        elif case == 'end' and self.verbose:
            print(f'{self.split_by_group} group processing complete!')
        else:
            pass   
