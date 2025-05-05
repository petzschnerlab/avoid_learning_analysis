#Import modules
from . import Master

#Pipeline class
class Pipeline(Master):

    """
    Class to run the SOMA project pipeline
    """

    def __init__(self):
        super().__init__()

    def run(self, **kwargs):

        '''
        Run the SOMA pipeline
        
        '''

        #Collect the parameters
        self.set_parameters(**kwargs)

        #Run the help
        if self.help:
            self.print_help()
            return None
        
        #Run the pipeline
        self.announce(case='start')
        self.load_data(file_path = self.file_path, file_name = self.file_name)
        if 'depression' in self.split_by_group_id:
            self.recode_depression()
        if 'pain' in self.split_by_group_id:
            self.exclude_pain(threshold=20)
        if self.check_data():
            self.process_data()
            self.run_tests()
            self.run_statistics()
            self.build_report(self.rscripts_path, self.load_stats, self.load_models)
            self.announce(case='end')