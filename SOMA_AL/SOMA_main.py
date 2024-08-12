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

    def __init__(self, print_filename=r'SOMA_AL/reports/SOMA_report.pdf'):
        super().__init__()

        self.print_filename = print_filename

    def run(self, file_path: str, file_name: list, split_by_group: str = 'pain'):

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
    file_name = [r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
    split_by_groups = ['pain', 'depression'] #'pain' or 'depression'

    for split_by_group in split_by_groups:
        SOMA = SOMAPipeline()
        SOMA.run(file_path=file_path, file_name=file_name, split_by_group=split_by_group) 

    #Debug stop
    print()