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

        #Modify codes
        self.data['group_code'] = self.data['group_code'].replace({0: 'no pain', 1: 'acute pain', 2: 'chronic pain'})
        self.data['symbol_L_value'] = self.data['symbol_L_name'].replace({'75R1': 4, '75R2': 4, '25R1': 3, '25R2': 3, '25P1': 2, '25P2': 2, '75P1': 1, '75P2': 1, 'Zero': 0})
        self.data['symbol_R_value'] = self.data['symbol_R_name'].replace({'75R1': 4, '75R2': 4, '25R1': 3, '25R2': 3, '25P1': 2, '25P2': 2, '75P1': 1, '75P2': 1, 'Zero': 0})

    def process_data(self):
        
        #Filter the learning and transfer data
        self.filter_learning_data()
        self.filter_transfer_data()

        #Compute demographics
        self.compute_demographics()
        self.compute_pain_scores()

        #Compute accuracy for learning data
        self.compute_accuracy()

        #Compute choice rate for transfer data
        self.compute_choice_rate()

    def save_processed_data(self):
        #Save the processed data to a new file
        self.data.to_csv(self.file.replace('.csv', '_processed.csv'))
        self.learning_data.to_csv(self.file.replace('.csv', '_learning_processed.csv'))
        self.transfer_data.to_csv(self.file.replace('.csv', '_transfer_processed.csv'))

    def filter_learning_data(self):

        #Filter data
        self.learning_data = self.data[self.data['trial_type'] == 'learning-trials'].reset_index(drop=True)

        #Create trial indices
        self.learning_data['trial_number'] = self.learning_data.groupby(['participant_id', 'context_val_name']).cumcount() + 1
        

    def filter_transfer_data(self):

        #Filter data
        self.transfer_data = self.data[self.data['trial_type'] == 'probe'].reset_index(drop=True)

        #Determine which symbol_n_value was chosen using the choice_made column where 1 = Right, 0 = Left in the transfer data
        self.transfer_data['symbol_chosen'] = self.transfer_data['symbol_L_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_chosen'] = self.transfer_data['symbol_R_value']
        self.transfer_data['symbol_ignored'] = self.transfer_data['symbol_R_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_ignored'] = self.transfer_data['symbol_L_value']

    def compute_accuracy(self):
        
        #Add computations to determine accuracy #TODO: THIS ONLY WORKS FOR LEARNING TRIALS
        self.learning_data['larger_value'] = (self.learning_data['symbol_R_value'] > self.learning_data['symbol_L_value']).astype(int) #1 = Right has larger value, 0 = Left has larger value
        self.learning_data['accuracy'] = (self.learning_data['larger_value'] == self.learning_data['choice_made']).astype(int)*100 #100 = Correct, 0 = Incorrect

    def compute_choice_rate(self):

        #Compute choice rates for each participant and symbol within each group
        choice_rate = pd.DataFrame(columns=['choice_rate'], index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=['group', 'participant', 'symbol']))
        for group in ['no pain', 'acute pain', 'chronic pain']:
            group_data = self.transfer_data[self.transfer_data['group_code'] == group]
            for participant in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant]
                for symbol in [0, 1, 2, 3, 4]:
                    symbol_chosen = participant_data[participant_data['symbol_chosen'] == symbol].shape[0]
                    symbol_ignored = participant_data[participant_data['symbol_ignored'] == symbol].shape[0]
                    symbol_choice_rate = symbol_chosen / (symbol_chosen + symbol_ignored) * 100

                    #Insert symbol_choice_rate into a new dataframe with index levels [group, participant, symbol]
                    choice_rate.loc[(group, participant, symbol), 'choice_rate'] = symbol_choice_rate

        self.choice_rate = choice_rate

    def compute_demographics(self):

        #Compute the demographics of the participants
        self.demographics = self.data.groupby(['group_code','participant_id'])[['age', 'sex']].first().reset_index()

        #Compute demographics statistics
        self.mean_age = self.demographics.groupby('group_code')['age'].mean()
        self.std_age = self.demographics.groupby('group_code')['age'].std()

        self.female_counts = self.demographics[self.demographics['sex'] == 'Female'].groupby('group_code')['participant_id'].nunique()
        self.male_counts = self.demographics[self.demographics['sex'] == 'Male'].groupby('group_code')['participant_id'].nunique()
        self.not_specified_counts = self.demographics[self.demographics['sex'] == 'Prefer not to say'].groupby('group_code')['participant_id'].nunique()
        self.not_specified_counts = self.not_specified_counts.reindex(['acute pain', 'chronic pain', 'no pain'], fill_value=0)

        #combine female counts, male counts, not specified into strings separated by slashes
        self.demo_sample_size = self.demographics.groupby('group_code')['participant_id'].nunique()
        self.demo_age = self.mean_age.round(2).astype(str) + ' (' + self.std_age.round(2).astype(str) + ')'
        self.demo_gender = self.female_counts.astype(str) + ' / ' + self.male_counts.astype(str) + ' / ' + self.not_specified_counts.astype(str)

        #Combine all demographics statistics into a single dataframe
        self.demographics_summary = pd.concat([self.demo_sample_size, self.demo_age, self.demo_gender], axis=1)
        self.demographics_summary.columns = ['Sample Size', 'Age', 'Gender (F/M/N)']
        self.demographics_summary = self.demographics_summary.reindex(['no pain', 'acute pain', 'chronic pain'])
        self.demographics_summary = self.demographics_summary.T

    #pandas create a summary of data using groupby
    def compute_pain_scores(self):

        self.mean_pain = self.data.groupby('group_code')[['intensity', 'unpleasant', 'interference']].mean()
        self.std_pain = self.data.groupby('group_code')[['intensity', 'unpleasant', 'interference']].std()
        self.pain_summary = self.mean_pain.round(2).astype(str) + ' (' + self.std_pain.round(2).astype(str) + ')'
        self.pain_summary = self.pain_summary.reindex(['no pain', 'acute pain', 'chronic pain'])
        self.pain_summary = self.pain_summary.T

    