#System settings
import sys
sys.dont_write_bytecode = True
import warnings

#Import modules
from helpers import SOMAMaster

class SOMAPipeline(SOMAMaster):

    """
    Class to run the SOMA project pipeline
    """

    def __init__(self, **kwargs):
        super().__init__()

        #Warning of unknown params
        accepted_params = ['author']
        for key in kwargs:
            if key not in accepted_params:
                warnings.warn(f'Unknown parameter {key} is being ignored', stacklevel=2)

        #Set input parameters
        self.author = kwargs.get('author', 'SOMA_Team')

        #Set backend parameters
        self.figure_count = 1
        self.table_count = 1

    def run(self, **kwargs):

        #Warning of unknown params
        accepted_params = ['file_path', 
                           'file_name', 
                           'print_filename', 
                           'split_by_group',
                           'accuracy_exclusion_threshold',
                           'RT_low_threshold',
                           'RT_high_threshold',
                           'rolling_mean',
                           'tests']
        for key in kwargs:
            if key not in accepted_params:
                warnings.warn(f'Unknown parameter {key} is being ignored', stacklevel=2)

        #Warning of missing required params
        required_params = ['file_path', 
                           'file_name']
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f'Missing required parameter {param}, which does not contain a default. Please provide a value for this parameter.')

        #Turn parameters into a kwargs dictionary #file_path: str, file_name: list, split_by_group: str = 'pain'
        self.file_path = kwargs.get('file_path', None)
        self.file_name = kwargs.get('file_name', None)
        self.print_filename = kwargs.get('print_filename', r'SOMA_AL/reports/SOMA_report.pdf')
        self.split_by_group = kwargs.get('split_by_group', 'pain')
        self.accuracy_exclusion_threshold = kwargs.get('accuracy_exclusion_threshold', 55)
        self.RT_low_threshold = kwargs.get('RT_low_threshold', 200)
        self.RT_high_threshold = kwargs.get('RT_high_threshold', 5000)
        self.rolling_mean = kwargs.get('rolling_mean', 5)
        self.tests = kwargs.get('tests', 'basic') #'basic' or 'extensive'
        
        #Set parameters
        self.split_by_group = split_by_group
        self.print_filename = self.print_filename.replace('.pdf', f'_{split_by_group}.pdf')
        self.group_code = 'group_code' if self.split_by_group == 'pain' else 'depression'
        if split_by_group == 'pain':
            self.group_labels = ['no pain', 'acute pain', 'chronic pain']
            self.group_labels_formatted = ['No\nPain', 'Acute\nPain', 'Chronic\nPain']
        else:
            self.group_labels = ['healthy', 'depressed']
            self.group_labels_formatted = ['Healthy', 'Depressed']

        #Load data
        self.load_data(file_path = file_path, file_name = file_name)
        
        if self.check_data():
            #Process data
            self.process_data()
            #SOMA_pipeline.save_processed_data()

            #Test code
            self.run_tests()

            #Report data
            self.build_report()

if __name__ == '__main__':

    file_path = r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific'
    #file_name = [r'v1a_avoid_pain\v1a_avoid_pain.csv', r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
    #file_name = [r'v1a_avoid_pain\v1a_avoid_pain.csv']
    file_name = [r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']

    #Set analyses dependent on whether there is only depression groups
    split_by_groups = ['pain'] if any("v1a" in s for s in file_name) else ['pain', 'depression'] #'pain' or 'depression'

    #Run the pipeline for each split_by_group
    for split_by_group in split_by_groups:
        
        #Create a dict of args
        kwargs = {'file_path': file_path,
            'file_name': file_name,
            'split_by_group': split_by_group,
            'tests': 'extensive'}

        SOMA_pipeline = SOMAPipeline(author='Chad C. Williams')
        SOMA_pipeline.run(**kwargs)

    #Debug stop
    print()