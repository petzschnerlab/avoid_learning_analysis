import os
import pandas as pd

#SOMAALPipeline class
class SOMAALPipeline:
    #Create a constructor method
    def __init__(self, print_filename=r'SOMA_AL/reports/SOMA_report.txt'):
        self.print_filename = print_filename

    #Create a decorator method
    def print_data(self, title, content):

        #Print to a file 
        with open(self.print_filename, 'a') as f:
            f.write('\n')
            f.write('==============================\n')
            f.write(title)
            f.write('\n')
            f.write('==============================\n')
            f.write('\n')
            if isinstance(content, list):
                self.print_subdata(content[0], content[1], f=f)
            else:
                if isinstance(content, pd.DataFrame):
                    f.write(content.to_string())
                else:
                    f.write(content)
            f.write('\n')

    def print_subdata(self, title: list, content, f):

        #Check if title and content are lists of equal length else return an error message
        if len(title) != len(content):
            ValueError('Length of title and content must be equal')

        #Iterate through the title and content to return a formatted string
        for content_name, item in zip(title, content):
            f.write(f'{content_name}: {item}')
            f.write('\n')

    #Create a method to load the data
    def load_data(self, file_path, file_name):
        #Create variables to store the file path and file name
        self.file_path = file_path
        self.file_name = file_name
        self.file = os.path.join(file_path, file_name)
        self.print_filename = f'{self.print_filename.replace(".txt","")}_{file_name.replace(".csv","")}.txt'

        #Load data
        self.data = pd.read_csv(self.file)

        #Modify data
        self.data = self.data.rename(columns = {'Unnamed: 0': 'participant_index'})
        self.data['group_code'] = self.data['group_code'].replace({0: 'No pain', 1: 'Acute pain', 2: 'Chronic pain'})

        #Summarize data
        self.summarize_data()
        self.groupby_summary('group_code')
    
    #Create a method to summarize the data
    def summarize_data(self):

        #Create a variable named summary and assign it the value of self.data.describe()
        self.data_summary = self.data.describe()

    #pandas create a summary of data using groupby
    def groupby_summary(self, groupby_column):

        self.grouped_summary = self.data.groupby(groupby_column)[['intensity', 'unpleasant', 'interference']].agg(['mean', 'std'])
        self.grouped_summary = self.grouped_summary.reindex(['No pain', 'Acute pain', 'Chronic pain'])
    
    def print_report(self):

        #Initiate the report
        with open(self.print_filename, 'w') as f:
            f.write(f'SOMA Report\nFile: {self.file_name}\n')
            f.write('\n')

        #Populate the report
        self.print_data('Column Names:', ', '.join(self.data.columns))
        self.print_data('Data Dimensions:', 
                        [['Rows', 'Columns', 'Number of Participants'],
                         [self.data.shape[0], self.data.shape[1], self.data['participant_index'].nunique()]])
        self.print_data('Data Head:', self.data.head())
        #self.print_data('Data Summary:', self.data_summary) #TODO: Removed because it collapses across all conditions and groups
        self.print_data('Grouped Summary of Pain:', self.grouped_summary)
    