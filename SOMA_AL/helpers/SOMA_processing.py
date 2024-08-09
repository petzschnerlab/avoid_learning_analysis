#Import modules
import os
import pandas as pd

#SOMAALPipeline class
class SOMAProcessing:

    """
    Class to hold processing functions for the SOMA project
    """
    #Create a method to load the data
    def load_data(self, file_path, file_name):
        #Create variables to store the file path and file name
        self.file_path = file_path
        self.file_name = file_name if isinstance(file_name, list) else [file_name]
        
        #check if file name is a list and if so, load each dataset and concatenate them, adding NAs if columns are missing
        if isinstance(file_name, list):
            data = []
            self.file = []
            for file in file_name:
                self.file.append(os.path.join(file_path, file))
                data.append(pd.read_csv(os.path.join(file_path, file)))

            #Drop any columns that do not exist in the first dataset and reorganize columns to match the first dataset
            for i in range(1, len(data)):
                data[i] = data[i].drop(columns = [col for col in data[i].columns if col not in data[0].columns])
                data[i] = data[i][data[0].columns]
            
            #Concatenate the data
            self.data = pd.concat(data, axis=0)
            file_name = '_AND_'.join([file.split('\\')[-1] for file in file_name])

        else:

            self.file = os.path.join(file_path, file_name)
            self.data = pd.read_csv(self.file)
            file_name = file_name.split('\\')[-1]

        #Modify the print filename to include the file name
        self.print_filename = f'{self.print_filename.replace(".pdf","")}_{file_name.replace(".csv","")}.pdf'

        #Modify unnamed column
        self.data = self.data.drop(columns = ['Unnamed: 0'])
    
    def process_data(self):
        #Create a dictionary to replace the group code with the group name
        self.data['group_code'] = self.data['group_code'].replace({0: 'no pain', 1: 'acute pain', 2: 'chronic pain'})

        #Add computations to determine accuracy #TODO: THIS ONLY WORKS FOR LEARNING TRIALS
        self.data['symbol_L_value'] = self.data['symbol_L_name'].replace({'75R1': 4, '75R2': 4, '25R1': 3, '25R2': 3, '25P1': 2, '25P2': 2, '75P1': 1, '75P2': 1, 'Zero': 0})
        self.data['symbol_R_value'] = self.data['symbol_R_name'].replace({'75R1': 4, '75R2': 4, '25R1': 3, '25R2': 3, '25P1': 2, '25P2': 2, '75P1': 1, '75P2': 1, 'Zero': 0})
        self.data['larger_value'] = (self.data['symbol_R_value'] > self.data['symbol_L_value']).astype(int) #1 = Right has larger value, 0 = Left has larger value
        self.data['accuracy'] = (self.data['larger_value'] == self.data['choice_made']).astype(int) #1 = Correct, 0 = Incorrect

        #Filter data
        self.learning_data = self.data[self.data['trial_type'] == 'learning-trials'].reset_index(drop=True)
        self.transfer_data = self.data[self.data['trial_type'] == 'probe'].reset_index(drop=True)

        #Create trial indices
        self.learning_data['trial_number'] = self.learning_data.groupby(['participant_id', 'context_val_name']).cumcount() + 1

        #Determine which symbol_n_value was chosen using the choice_made column where 1 = Right, 0 = Left in the transfer data
        self.transfer_data['symbol_chosen'] = self.transfer_data['symbol_L_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_chosen'] = self.transfer_data['symbol_R_value']
        self.transfer_data['symbol_ignored'] = self.transfer_data['symbol_R_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_ignored'] = self.transfer_data['symbol_L_value']

    def save_processed_data(self):
        #Save the processed data to a new file
        self.data.to_csv(self.file.replace('.csv', '_processed.csv'))
        self.learning_data.to_csv(self.file.replace('.csv', '_learning_processed.csv'))
        self.transfer_data.to_csv(self.file.replace('.csv', '_transfer_processed.csv'))

    #Create a method to summarize the data
    def summarize_data(self):

        #Create a variable named summary and assign it the value of self.data.describe()
        self.data_summary = self.data.describe()

    #pandas create a summary of data using groupby
    def groupby_summary(self, groupby_column):

        self.grouped_summary = self.data.groupby(groupby_column)[['intensity', 'unpleasant', 'interference']].agg(['mean', 'std'])
        self.grouped_summary = self.grouped_summary.reindex(['no pain', 'acute pain', 'chronic pain'])
    