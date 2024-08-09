#System settings
import sys
sys.dont_write_bytecode = True

#Import modules
from helpers import SOMAMaster

class SOMAPipeline(SOMAMaster):

    def __init__(self, print_filename=r'SOMA_AL/reports/SOMA_report.pdf'):
        super().__init__()

        self.print_filename = print_filename

    def run(self, file_path: str, file_name: list):

        #Load data
        self.load_data(file_path = file_path, file_name = file_name)
        
        #Process data
        self.process_data()
        #SOMA_pipeline.save_processed_data()

        #Compute summary statistics
        self.summarize_data()
        self.groupby_summary('group_code')

        #Test code
        self.run_tests()

        #Report data
        self.build_report()

if __name__ == '__main__':

    file_path = r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific'
    file_names = [r'v1a_avoid_pain\v1a_avoid_pain.csv', r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
    #file_names = [r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']

    SOMA = SOMAPipeline()
    SOMA.run(file_path, file_names) 