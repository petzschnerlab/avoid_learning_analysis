class Help:

    """
    Class to display help information for the package. 
    """

    def __init__(self):

        """
        Initialize the Help class with parameter descriptions.
        This class provides an overview of the package and details about each parameter that can be set when running the package.

        """
        self.pipeline_parameters = [
            'help',
        ]

        self.run_parameters = [
            'help',
            'author',
            'print_filename',
            'file_path',
            'file_name',
            'dataset',
            'split_by_group',
            'split_by_group_id',
            'accuracy_exclusion_threshold',
            'RT_low_threshold',
            'RT_high_threshold',
            'rscripts_path',
            'pain_cutoff',
            'depression_cutoff',
            'rolling_mean',
            'load_stats',
            'load_posthocs',
            'load_models',
            'hide_stats',
            'hide_posthocs',
            'verbose',
        ]
    
        self.parameter_descriptions = {
            'help':
                ['Prints the help information for the package, including an overview and parameter descriptions.',
                'bool',
                False],

            'author':
                ['Author of the report, which will be displayed on the report.',
                'str',
                'PEAC_Team'],

            'rscripts_path':
                [('Path to the R executible file, which is used when running GLMMs. '
                  'The reason this is done in R is because the statsmodels package in Python does not provide factor level p-values for (generalized) linear mixed effects models. '
                  'This is tricky because you need to have R and a few packages (lme4, lmerTest, car, afex, and emmeans) installed. '
                  'This is worth looking into further, as there might be a parameter I have overlooked, or else there could be a different package that fits our needs.'),
                'str',
                None],

            'file_path':
                [('Path to the data file(s) to be loaded. This is a required parameter. ' 
                  'From this path, you can load several different files using the file_name parameter. '), 
                'str',
                None],

            'file_name': 
                [('Name of the file(s) to be loaded. This is a required parameter. '
                  'These filenames should be relative to the file_path parameter. '
                  'You can load multiple files by providing a list of file names or a single file name as a string. '
                  'You can add further path information here if your data splits at the point of file_path. '
                  'For example, file_path = "path/to/data" and file_name = '
                  '["subfolder1/data1.csv", "subfolder2/data2.csv"] will load two files from different subfolders.'),
                 'list[str] | str',
                 None],

            'print_filename':
                [('This package generates a report including some main demographics, statistics, and figures. '
                  'The report will be saved at the provided location, originating from the root directory of the repo. '
                  'This will overwrite any existing report with the same name, but if this report is open, '
                  'it will add -N to the report name where N increases from 1 until a unique name is found. '), 
                'str',
                r'AL/reports/SOMA_report.pdf'],
            
            'split_by_group':
                [('Split the data by clinical group, options are \'pain\' and \'depression\'. '
                  'Unfortunately, somewhere along the way, I stopped testing the depression group, so it crashes. '
                  'A lot of functionality is in place for this group, but lots of stats/plots were added after I stopped testing it. '
                  'Moreover, the loading of model data was also not tested for the depression group. '
                  'So, future development can bring this back to life, potentially without much effort. '
                  'But, do not use it for now. '),
                'str',
                'pain'],

            'split_by_group_id':
                [('This ID is used as a unique identifier for the analysis. '
                  'That way, you can run the pain analysis multiple times (with different parameters) '
                  'without overwriting the previous results. '
                  'This is designed around statistics, which saves data and results using this ID. '
                  'If you do not add this parameter, it will default to the value of the split_by_group parameter. '),
                'str',
                None],

            'dataset': 
                [('This is a description of the dataset(s) in the report. '
                  'It allows for you to provide a brief overview of the dataset(s) used in the analysis, '
                  'if you like.'),
                'str',
                ''],

            'pain_cutoff':
                [('A threshold used as exclusion criteria for participant pain group IDs. '
                  'Threshold considers the composite pain score (average of intensity, unpleasantness, and interference) '
                  'This score ranges from 0 to 10, where 0 is no pain and 10 is the worst pain imaginable. '
                  'The cutoff is used wherein any participant in the no pain group that exceeds this threshold will be excluded from the analysis. '
                  'And any participant in the pain groups (acute, chronic) that falls below this threshold will be excluded from the analysis. '
                  'If set to None, no participants will be excluded based on pain. '),
                'int',
                2],


            'depression_cutoff':
                [('A threshold used to classify participants into depression groups. '
                  'This threshold considers the PHQ-8 score, which ranges from 0 to 24. '
                  'The cutoff is used to classify participants into the depression group '
                  'if their score is equal to or above this threshold. '
                  'The data (which is determined using the avoid_learning_data_extraction repo) will already have participants classified into the depression group, '
                  'but this parameter overrides that classification (because the data incorrectly used a cutoff of > 10).'),
                'int',
                10],

            'accuracy_exclusion_threshold':
                [('Threshold for excluding participants based on their accuracy (%) in the learning phase. '
                  'Specifically, this is the accuracy in the last quarter of the learning phase. '
                  'Anyone below this threshold will be excluded from the analysis. '
                  'The number of people excluded will be indicated in the report. '),
            'int',
            70],
            
            
            'RT_low_threshold':
                ['Lower threshold for excluding trials based on reaction times (ms).',
                'int',
                200],

            'RT_high_threshold':
                ['Upper threshold for excluding trials based on reaction times (ms).',
                'int',
                5000],

            'rolling_mean':
                [('This parameter determines how many trials to average across the learning phase for plotting the learning curves. '
                  'This functionality is actually no longer used in the current implementation, because learning curves are now plotted using the '
                  'binned trials data (early, mid-early, mid-late, late). '
                  'In the plotting class, the plot_learning_curves function has a binned_trial parameter, which can manually be set to False. '
                  'If it is set to False (which requires changing the code), then the rolling mean will be used. '),
                'int',
                5],

            'load_stats':
                [('The GLMMs take a long time to run, so the results are saved in the backend. '
                  'If this parameter is set to True, the results will be loaded from the backend, '
                  'instead of running the GLMMs every time this has been run. '
                  'If this is set to True, but the GLMMs have not been run, it will run the GLMMs anyhow. '
                  'Make sure to re-run these stats anytime changes are made to the data. '),
                'bool',
                False],

            'load_posthocs':
                [('The posthoc tests can take a long time to add to the report, so the results are saved in the backend. '
                  'Specifically, tables are turned into png images (which is what takes long), and then uploaded to the report. '
                  'If this parameter is set to True, the results will be loaded from the backend, and will skip saving to png. '
                  'Otherwise, it will take the time to save posthocs to png images. '
                  'If your posthoc stats change, you will need to set this to False for the changes to be reflected in the report.'),
                'bool',
                False],

            'load_models':
                [('If set to True, the models will be loaded from the backend. '
                  'This is useful if you have run the avoid_learning_rl_models package and want to include the model results in the report. '
                  'If set to False, the models will not be loaded, and the report will not include any model information. '
                  'Incorporating data from the avoid_learning_rl_models requires a few steps. '
                  'You need to first run the analysis with this repo, which will generate data that the avoid_learning_rl_models package can use. '
                  'This data will be saved in AL/data/ as pain_learning_processed.csv and pain_transfer_processed.csv. '
                  'You then have to manually copy these files to the avoid_learning_rl_models/RL/data/ directory. '
                  'After that, you can run the avoid_learning_rl_models package, which will generate the models, and a directory called \'RL/modelling\''
                  'This directory can be manually copied to the AL directory, as \'AL/modelling\'. '
                  'Then, you can set this parameter to True, and the models will be loaded from the backend. '
                  'Sorry this is a bit complicated, with manual steps, but I felt keeping these repos as standalones were better than merging them. '
                  ),
                'bool',
                False],

            'hide_stats': 
                [('If set to True, the statistics will not be hidden in the report. '
                  'They will exist but their values will be replaced with \'hidden\'. '
                  'I included this functionality because I believe stats should only be observed once, to mitigate p-hacking. '
                  'This way, you can hide the stats but still see all other information, such as report details and plots. '
                  'If you want to see the stats, set this to False. '),
                'bool',
                False],

            'hide_posthocs':
                [('In these analyses, we end up with a lot of posthoc tests, which can clutter the report. '
                  'It literally adds 40+ pages of tables to the report. '
                  'This takes a long time to generate, and is not always necessary. '
                  'Furthermore, it makes the report very large, which is not ideal for sharing. '
                  'If this parameter is set to True, the posthoc tests will not be included in the report, '
                  'saving time and space.'),
                'bool',
                False],

            'verbose':
                [('If set to True, the pipeline will print additional information to the console during execution. '
                  'This can be useful for debugging or understanding the flow of the pipeline. '
                  'If set to False, the pipeline will run more silently without printing additional information.'),
                'bool',
                False]
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
        
        overview = """The Avoid Learning Analysis package is a tool to analyze the data from the SOMA avoidance learning study.
        The package is designed to be run as a function with parameters that can be set to customize the analysis.
        The package will generate a report with the main results of the analysis.
        The package requires two parameters to be set: file_path and file_name.
        In addition, the package has a number of optional parameters that can be set to customize the analysis, which will be described below.
        This package is a standalone package but can also be used with the companion package avoid_learning_rl_models.
        If modeling has been run using the companion package, the load_models parameter can be set to True, which will load the models from the backend and include them in the report.
        
        The easiest implementation of this package is:
            from helpers.pipeline import Pipeline
            
            kwargs = {
                'file_path': 'path/to/data',
                'file_name': ['subfolder1/data1.csv', 'subfolder2/data2.csv']}
            
            pipeline = Pipeline()
            pipeline.run(**kwargs)

        Please note: The docstrings within the classes and methods of this package are mostly built using AI. They should be alright, but there may be errors
        However, this help function was written by hand and should be accurate. If there is any conflict between the two, believe the help function.
        If the help function is not clear, or wrong, sorry! Feel free to open an issue on the GitHub repository or make changes and submit a pull request.
        """

        print(overview)

    def print_parameters(self) -> None:

        """
        Print the parameter descriptions.
                Returns
        -------
        None
        """
        print('\n==============================================================')
        print('==============================================================')
        print('PIPELINE OBJECT PARAMETERS')
        print('-------------------------\n')
        print('from helpers.pipeline import Pipeline')
        print('pipeline = Pipeline(**params)')
        print('==============================================================\n')
        for param in self.pipeline_parameters:
            desc = self.parameter_descriptions[param]
            print(f'{param}\n  DESCRIPTION: {desc[0]}\n  TYPE: {desc[1]}\n  DEFAULT: {desc[2]}\n')

        print('\n==============================================================')
        print('RUN PARAMETERS')
        print('-------------------\n')
        print('from helpers.pipeline import Pipeline')
        print('pipeline = Pipeline()')
        print('pipeline.run(**params)')
        print('==============================================================\n')
        for param in self.run_parameters:
            desc = self.parameter_descriptions[param]
            print(f'{param}\n  DESCRIPTION: {desc[0]}\n  TYPE: {desc[1]}\n  DEFAULT: {desc[2]}\n')

        print('==============================================================\n')
