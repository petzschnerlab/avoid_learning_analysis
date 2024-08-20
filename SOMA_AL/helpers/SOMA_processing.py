#Import modules
import os
import warnings
import pandas as pd
import numpy as np
pd.set_option("future.no_silent_downcasting", True)

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

        #Modify unnamed column
        self.data = self.data.drop(columns = ['Unnamed: 0'])
        
        #Determine number of participants
        self.participants_original = self.data['participant_id'].nunique()

        #Modify codes
        if self.split_by_group == 'pain':
            self.data[self.group_code] = self.data[self.group_code].replace({0: 'no pain', 1: 'acute pain', 2: 'chronic pain'})
        else:
            self.data[self.group_code] = self.data[self.group_code].replace({0: 'healthy', 1: 'depressed'})

        self.data['symbol_L_value'] = self.data['symbol_L_name'].replace({'75R1': 4, '75R2': 4, '25R1': 3, '25R2': 3, '25P1': 2, '25P2': 2, '75P1': 1, '75P2': 1, 'Zero': 0})
        self.data['symbol_R_value'] = self.data['symbol_R_name'].replace({'75R1': 4, '75R2': 4, '25R1': 3, '25R2': 3, '25P1': 2, '25P2': 2, '75P1': 1, '75P2': 1, 'Zero': 0})
        self.data['neutral_values'] = ((self.data['symbol_L_value'] == 3) | (self.data['symbol_L_value'] == 2)) & ((self.data['symbol_R_value'] == 3) | (self.data['symbol_R_value'] == 2))

    def check_data(self):
        if 'depression' not in self.data.columns and self.split_by_group == 'depression':
            warnings.warn('No depression scores found in the data. Skipping depression score computation.')
            return False

        return True

    def process_data(self):
        
        #Filter the learning and transfer data
        self.filter_learning_data()
        self.filter_transfer_data()

        #Compute accuracy for learning data
        self.compute_accuracy()

        #Exclude participants with low accuracy and trials with low reaction times
        self.exclude_low_accuracy(self.accuracy_exclusion_threshold)
        self.exclude_low_rt(self.RT_low_threshold, self.RT_high_threshold)

        #Compute demographics and scores
        self.compute_demographics()
        self.compute_pain_scores()
        self.compute_depression_scores()

        #Compute choice rate for transfer data
        self.compute_choice_rate()
        self.compute_choice_rate(neutral=True)

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
        self.transfer_data = self.transfer_data[self.transfer_data['symbol_L_value'] != self.transfer_data['symbol_R_value']] #Remove trials where the same valued symbol was presented together

        #Determine which symbol_n_value was chosen using the choice_made column where 1 = Right, 0 = Left in the transfer data
        self.transfer_data['symbol_chosen'] = self.transfer_data['symbol_L_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_chosen'] = self.transfer_data['symbol_R_value']
        self.transfer_data['symbol_ignored'] = self.transfer_data['symbol_R_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_ignored'] = self.transfer_data['symbol_L_value']

    def compute_accuracy(self):
        
        #Add computations to determine accuracy
        self.learning_data['larger_value'] = (self.learning_data['symbol_R_value'] > self.learning_data['symbol_L_value']).astype(int) #1 = Right has larger value, 0 = Left has larger value
        self.learning_data['accuracy'] = (self.learning_data['larger_value'] == self.learning_data['choice_made']).astype(int)*100 #100 = Correct, 0 = Incorrect

    def exclude_low_accuracy(self, threshold=60):
        #Compute accuracy for each participant
        accuracy = pd.DataFrame(columns=['accuracy'], index=pd.MultiIndex(levels=[[]], codes=[[]], names=['participant']))
        for participant in self.learning_data['participant_id'].unique():
            participant_data = self.learning_data[self.learning_data['participant_id'] == participant]
            accuracy_rate = participant_data['accuracy'].mean()
            accuracy.loc[participant, 'accuracy'] = accuracy_rate

        #Find participants with accuracy less than 60%
        low_accuracy = accuracy[accuracy['accuracy'] < threshold].reset_index()

        #Remove participants with accuracy less than 60%
        self.data = self.data[~self.data['participant_id'].isin(low_accuracy['participant'])]
        self.learning_data = self.learning_data[~self.learning_data['participant_id'].isin(low_accuracy['participant'])]
        self.transfer_data = self.transfer_data[~self.transfer_data['participant_id'].isin(low_accuracy['participant'])]

        #Track number of participants excluded
        self.participants_excluded_accuracy = len(low_accuracy)
        self.accuracy_threshold = threshold

    def exclude_low_rt(self, low_threshold=200, high_threshold=5000):

        self.learning_data['excluded_rt'] = (self.learning_data['rt'] < low_threshold) | (self.learning_data['rt'] > high_threshold)
        self.learning_data.loc[self.learning_data['excluded_rt'] == True, 'rt'] = np.nan
        self.learning_data.loc[self.learning_data['excluded_rt'] == True, 'accuracy'] = np.nan

        self.transfer_data['excluded_rt'] = (self.transfer_data['rt'] < low_threshold) | (self.transfer_data['rt'] > high_threshold)
        self.transfer_data.loc[self.transfer_data['excluded_rt'] == True, 'rt'] = np.nan
        self.transfer_data.loc[self.transfer_data['excluded_rt'] == True, 'symbol_chosen'] = np.nan
        self.transfer_data.loc[self.transfer_data['excluded_rt'] == True, 'symbol_ignored'] = np.nan

        #Track number of participants excluded
        excluded_count = (self.learning_data['excluded_rt'].sum() + self.transfer_data['excluded_rt'].sum())
        total_trials = (self.learning_data.shape[0] + self.transfer_data.shape[0])
        self.trials_excluded_rt = excluded_count/total_trials * 100

    def compute_choice_rate(self, neutral = False):

        '''
        Neutral: Processing specific to the 25R and 25P symbols being compared
        '''

        data = self.transfer_data if not neutral else self.transfer_data[self.transfer_data['neutral_values']]

        #Compute choice rates for each participant and symbol within each group
        choice_rate = pd.DataFrame(columns=['choice_rate'], index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=['group', 'participant', 'symbol']))
        for group in self.group_labels:
            group_data = data[data[self.group_code] == group]
            for participant in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant]
                symbols = [0, 1, 2, 3, 4] if not neutral else [2, 3]
                for symbol in symbols:
                    symbol_chosen = participant_data[participant_data['symbol_chosen'] == symbol].shape[0]
                    symbol_ignored = participant_data[participant_data['symbol_ignored'] == symbol].shape[0]
                    symbol_choice_rate = symbol_chosen / (symbol_chosen + symbol_ignored) * 100

                    #Insert symbol_choice_rate into a new dataframe with index levels [group, participant, symbol]
                    choice_rate.loc[(group, participant, symbol), 'choice_rate'] = symbol_choice_rate

        if not neutral:
            self.choice_rate = choice_rate
        else:
            choice_rate = choice_rate.reset_index()
            choice_rate = choice_rate[choice_rate['symbol'] == 3] #Choose rewarding 
            choice_rate = choice_rate.set_index(['group', 'participant', 'symbol'])
            self.neutral_choice_rate = choice_rate

    def compute_demographics(self):

        #Compute the demographics of the participants
        self.demographics = self.data.groupby([self.group_code,'participant_id'])[['age', 'sex']].first().reset_index()

        #Compute demographics statistics
        self.mean_age = self.demographics.groupby(self.group_code)['age'].mean()
        self.std_age = self.demographics.groupby(self.group_code)['age'].std()

        self.female_counts = self.demographics[self.demographics['sex'] == 'Female'].groupby(self.group_code)['participant_id'].nunique()
        self.male_counts = self.demographics[self.demographics['sex'] == 'Male'].groupby(self.group_code)['participant_id'].nunique()
        self.not_specified_counts = self.demographics[self.demographics['sex'] == 'Prefer not to say'].groupby(self.group_code)['participant_id'].nunique()
        self.not_specified_counts = self.not_specified_counts.reindex(self.group_labels, fill_value=0)

        #combine female counts, male counts, not specified into strings separated by slashes
        self.demo_sample_size = self.demographics.groupby(self.group_code)['participant_id'].nunique()
        self.demo_age = self.mean_age.round(2).astype(str) + ' (' + self.std_age.round(2).astype(str) + ')'
        self.demo_gender = self.female_counts.astype(str) + ' / ' + self.male_counts.astype(str) + ' / ' + self.not_specified_counts.astype(str)

        #Combine all demographics statistics into a single dataframe
        self.demographics_summary = pd.concat([self.demo_sample_size, self.demo_age, self.demo_gender], axis=1)
        self.demographics_summary.columns = ['Sample Size', 'Age', 'Gender (F/M/N)']
        self.demographics_summary = self.demographics_summary.reindex(self.group_labels)
        self.demographics_summary = self.demographics_summary.T

    #Compute pain scores
    def compute_pain_scores(self):

        self.pain_scores = self.data.groupby([self.group_code, 'participant_id'])[['intensity', 'unpleasant', 'interference']].first().reset_index()
        self.mean_pain = self.data.groupby(self.group_code)[['intensity', 'unpleasant', 'interference']].mean()
        self.std_pain = self.data.groupby(self.group_code)[['intensity', 'unpleasant', 'interference']].std()
        self.pain_summary = self.mean_pain.round(2).astype(str) + ' (' + self.std_pain.round(2).astype(str) + ')'
        self.pain_summary = self.pain_summary.reindex(self.group_labels)
        self.pain_summary = self.pain_summary.T

    #Compute depression scores
    def compute_depression_scores(self):
        
        #Check if PHQ8 is in the data
        if 'PHQ8' in self.data.columns:
            self.depression_scores = self.data.groupby([self.group_code, 'participant_id'])['PHQ8'].first().reset_index()
            self.mean_depression = self.data.groupby(self.group_code)['PHQ8'].mean()
            self.std_depression = self.data.groupby(self.group_code)['PHQ8'].std()
            self.depression_summary = self.mean_depression.round(2).astype(str) + ' (' + self.std_depression.round(2).astype(str) + ')'
            self.depression_summary = self.depression_summary.reindex(self.group_labels)
            self.depression_summary = self.depression_summary.to_frame().T
        else:
            self.depression_scores = None
            self.depression_summary = None