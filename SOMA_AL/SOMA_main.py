#System settings
import sys
sys.dont_write_bytecode = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Import modules
from helpers import Master

#Pipeline class
class SOMAPipeline(Master):

    """
    Class to run the SOMA project pipeline
    """

    def __init__(self):
        super().__init__()     

    def run(self, **kwargs):

        #Run the pipeline
        self.set_parameters(**kwargs)
        self.announce(case='start')
        self.load_data(file_path = self.file_path, file_name = self.file_name)
        if 'depression' in self.split_by_group:
            self.recode_depression()
        if self.check_data():
            self.process_data()
            self.run_tests()
            self.run_statistics()
            self.build_report()
            self.announce(case='end')
                

if __name__ == '__main__':

    #Meta parameters
    author = 'Chad C. Williams' 

    #File parameters
    file_path = r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific'
    rscripts_path = 'C:/Program Files/R/R-4.4.1/bin/x64/Rscript'

    #Grouping parameters
    depression_cutoff = 10

    #Exclusion parameters
    accuracy_exclusion_threshold = 55
    RT_low_threshold = 200
    RT_high_threshold = 5000

    #Plotting parameters
    rolling_mean = 5 #None or int

    #Test parameters
    tests = 'basic' #'basic' or 'extensive'
    test_rolling_mean = 5
    test_context_type = 'context' #'context' or 'symbol'

    #Other parameters
    hide_stats = False
    load_stats = True
    verbose = True

    #Run the pipeline for each split_by_group
    for split_by_group in ['pain', 'depression']:

        if split_by_group == 'pain':
            file_name = [r'v1a_avoid_pain\v1a_avoid_pain.csv', r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
        elif split_by_group == 'depression':
            file_name = [r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
        
        #Create a dict of args
        kwargs = {'author': author,
                  'rscripts_path': rscripts_path,
                  'file_path': file_path,
                  'file_name': file_name,
                  'split_by_group': split_by_group,
                  'depression_cutoff': depression_cutoff,
                  'rolling_mean': rolling_mean,
                  'accuracy_exclusion_threshold': accuracy_exclusion_threshold,
                  'RT_low_threshold': RT_low_threshold,
                  'RT_high_threshold': RT_high_threshold,
                  'tests': tests,
                  'test_rolling_mean': test_rolling_mean,
                  'test_context_type': test_context_type,
                  'hide_stats': hide_stats,
                  'load_stats': load_stats,
                  'verbose': verbose}
        
        #Run the pipeline
        SOMA_pipeline = SOMAPipeline()
        SOMA_pipeline.run(**kwargs)

    #Debug stop
    print()