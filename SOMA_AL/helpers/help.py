class Help:

    """
    Class to display help information for the package. 
    """

    def __init__(self):
    
        self.parameter_descriptions = {
            'help': ['Print the help information.', 'bool', False],
            'author': ['Author of the package.', 'str', 'SOMA Team'],
            'rscripts_path': ['Path to the R executible file, which is used when running GLMMs.', 'str', None],
            'file_path': ['Path to the data file(s) to be loaded. This is a required parameter.', 'str', None],
            'file_name': ['Name of the file(s) to be loaded. This is a required parameter.', 'list[str] | str', None],
            'print_filename': ['The report filename.', 'str', r'SOMA_AL/reports/SOMA_report.pdf'],
            'split_by_group': ['Split the data by group [depression, pain].', 'str', 'pain'],
            'split_by_group_id': ['Analysis ID, which is used in the backend to save the statistics so you do not need to rerun them every time.', 'str', 'pain'],
            'dataset': ['Title to describe the dataset(s).', 'str', ''],
            'covariate': ['Covariate to use in GLMMs.', 'str', None],
            'depression_cutoff': ['Cutoff PQH8 score to be considered part of the depression group.', 'int', 10],
            'accuracy_exclusion_threshold': ['Threshold for excluding participants based on their accuracy (%) across the task.', 'int', 55],
            'RT_low_threshold': ['Lower threshold for excluding trials based on reaction times (ms).', 'int', 200],
            'RT_high_threshold': ['Upper threshold for excluding trials based on reaction times (ms).', 'int', 5000],
            'rolling_mean': ['Number of trials to use in the rolling mean for plotting learning curves', 'int', 5],
            'tests': ['Type of tests to run [basic, extensive].', 'str', 'basic'],
            'test_rolling_mean': ['Number of trials to use in the rolling mean for the tests learning curves.', 'int', None],
            'test_context_type': ['Context type to use in the tests learning curves.', 'str', 'context'],
            'hide_stats': ['Hide the statistics in the report.', 'bool', False],
            'load_stats': ['Load the statistics from the backend if pre-run.', 'bool', False],
            'hide_posthocs': ['Hide the posthocs in the report.', 'bool', False],
            'verbose': ['Print verbose output.', 'bool', False]
        }

    def print_help(self) -> None:
        
        """
        Print the help information.
        """

        self.print_overview()
        self.print_parameters()

    def print_overview(self) -> None:

        """
        Print the package overview.
        """
        
        overview = """The SOMA_AL package is a tool to analyze the data from the SOMA avoidance learning study.
        The package is designed to be run as a function with parameters that can be set to customize the analysis.
        The package will generate a report with the results of the analysis.
        The package requires two parameters to be set: file_path and file_name.
        In addition, the package has a number of optional parameters that can be set to customize the analysis, which will be described below.

        The easiest implementation of this package is:
        from SOMA_Main import SOMAPipeline
        kwargs = {'file_path': 'path_to_data', 'file_name': 'data.csv'}
        soma_pipeline = SOMAPipeline()
        soma_pipeline.run(**kwargs)
        """
        print(overview)

    def print_parameters(self) -> None:

        """
        Print the parameter descriptions.
        """

        print('SOMA_AL package help information:')
        print('The following parameters can be set when running the package:')
        print('-------------------------------------------------------------')

        for param in self.parameter_descriptions:
            print(param)
            print('Description:', self.parameter_descriptions[param][0])
            print('Type:', self.parameter_descriptions[param][1])
            print('Default:', self.parameter_descriptions[param][2])
            print('')

        print('-------------------------------------------------------------')