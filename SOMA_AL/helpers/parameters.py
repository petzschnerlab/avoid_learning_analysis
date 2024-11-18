#Import modules
import warnings
import os

class Parameters:

    """
    Class to set parameters for the SOMA pipeline
    """

    #Project organization
    def set_parameters(self, **kwargs):
        
        """
        Assigns parameters to the SOMA pipeline class
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
                           'covariate',
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
                           'verbose']
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
        self.repo_directory = os.path.dirname(os.path.realpath(__file__)).split('SOMA_AL')[0]

        #Unpack parameters
        self.kwargs = kwargs
        self.help = kwargs.get('help', False)
        self.author = kwargs.get('author', 'SOMA_Team')
        self.rscripts_path = kwargs.get('rscripts_path', None)
        self.file_path = kwargs.get('file_path', None)
        self.file_name = kwargs.get('file_name', None)
        self.dataset = kwargs.get('dataset', '')
        self.print_filename = kwargs.get('print_filename', r'SOMA_AL/reports/SOMA_report.pdf')
        self.split_by_group = kwargs.get('split_by_group', 'pain')
        self.split_by_group_id = kwargs.get('split_by_group_id', self.split_by_group)
        self.covariate = kwargs.get('covariate', None)
        self.depression_cutoff = kwargs.get('depression_cutoff', 10)
        self.accuracy_exclusion_threshold = kwargs.get('accuracy_exclusion_threshold', 55)
        self.RT_low_threshold = kwargs.get('RT_low_threshold', 200)
        self.RT_high_threshold = kwargs.get('RT_high_threshold', 5000)
        self.rolling_mean = kwargs.get('rolling_mean', 5)
        self.tests = kwargs.get('tests', 'basic') #'basic' or 'extensive'
        self.test_rolling_mean = kwargs.get('test_rolling_mean', None)
        self.test_context_type = kwargs.get('test_context_type', 'context')
        self.hide_stats = kwargs.get('hide_stats', False)
        self.load_stats = kwargs.get('load_stats', False)
        self.hide_posthocs = kwargs.get('hide_posthocs', False)
        self.verbose = kwargs.get('verbose', False)
        
        #Format parameters
        self.split_by_group = self.split_by_group if ',' not in self.split_by_group else self.split_by_group
        self.hide_posthocs = True if self.load_stats == False else self.hide_posthocs
        self.print_filename = self.print_filename.replace('.pdf', f'_{self.split_by_group_id}.pdf')
        self.group_code = 'group_code' if self.split_by_group == 'pain' else 'depression'
        if self.split_by_group == 'pain':
            self.group_labels = ['no pain', 'acute pain', 'chronic pain']
            self.group_labels_formatted = ['No\nPain', 'Acute\nPain', 'Chronic\nPain']
        else:
            self.group_labels = ['healthy', 'depressed']
            self.group_labels_formatted = ['Healthy', 'Depressed']

    #Print parameters
    def announce(self, case='start'):
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
