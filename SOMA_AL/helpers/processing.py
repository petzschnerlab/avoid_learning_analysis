#Import modules
import os
import warnings
import pandas as pd
import numpy as np
pd.set_option("future.no_silent_downcasting", True)

#SOMAALPipeline class
class Processing:

    """
    Class to hold processing functions for the SOMA project
    """

    #Data loading and checks
    def load_data(self, file_path: str, file_name: str) -> None:

        """
        Function to load the data from a csv file and store it in a pandas dataframe

        Parameters:
        -----------
        file_path : str
            The path to the file
        file_name : str 
            The name of the file

        Returns (Internal)
        ------------------
        self.file_path : str
            The path to the file
        self.file_name : str
            The name of the file
        self.data : pd.DataFrame
            The data loaded from the file
        self.participants_original : int
            The number of participants in the original data
        """
        
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

    def check_data(self) -> bool:

        """
        Function to check to see if depression data exists in dataframe if user is analyzing by depression

        Returns
        -------
        bool
            Whether the data is valid
        """

        if 'depression' not in self.data.columns and self.split_by_group == 'depression':
            warnings.warn('No depression scores found in the data. Skipping depression score computation.')
            return False

        return True
    
    #Data processing
    def process_data(self) -> None:

        """
        Main function to process the data for analysis
        """
        
        #Filter the learning and transfer data
        self.filter_learning_data()
        self.filter_transfer_data()

        #Compute accuracy
        self.compute_accuracy()
        self.compute_choice_rate() #Just for accuracy exclusion

        #Exclude participants with low accuracy
        self.exclude_low_accuracy(self.accuracy_exclusion_threshold)

        #Save processed data for RL modelling
        self.save_processed_data()

        #Exclude trials with low/high reaction times
        self.exclude_low_rt(self.RT_low_threshold, self.RT_high_threshold)

        #Compute metrics
        self.compute_learning_averages()
        self.compute_choice_rate()
        self.compute_choice_rate(neutral=True)
        self.compute_valence_bias()

        #Compute demographics and scores
        self.compute_demographics()
        self.compute_pain_scores()
        self.compute_depression_scores()

        #Average the data
        self.average_data()
    
    def combine_columns(self, x: pd.Series) -> str:

        """
        Function to combine the symbol_L_value and symbol_R_value columns into a single column

        Parameters
        ----------
        x : pd.Series
            The row of data to manipulate

        Returns
        -------
        column : str
            The combined column
        """

        if x['symbol_L_value'] > x['symbol_R_value']:
            column = str(x['symbol_L_value']) + '_' + str(x['symbol_R_value'])
        else:
            column = str(x['symbol_R_value']) + '_' + str(x['symbol_L_value'])

        return column
    
    def recode_depression(self) -> None:

        """
        Function to recode the depression scores into a binary variable

        Returns (Internal)
        ------------------
        self.data : pd.DataFrame
            The data with the depression scores recoded
        """

        self.data['depression'] = (self.data['PHQ8'] >= self.depression_cutoff).astype(int)
        self.data['depression'] = self.data['depression'].replace({0: 'healthy', 1: 'depressed'})

    def recode_pain(self) -> None:

        
        """
        Function to recode the pain scores into a binary variable

        Returns (Internal)
        ------------------
        self.data : pd.DataFrame
            The data with the pain scores recoded
        """

        #Create new group codes based on composite_pain scores
        self.data['composite_pain'] = self.data.apply(lambda x: np.mean((x['intensity'], x['unpleasant'], x['interference'])), axis=1)
        self.data['group_code'] = self.data.apply(lambda x: 'no pain' if x['composite_pain'] < 20 else ('acute pain' if (x['composite_pain'] >= 20) & (x['composite_pain'] <= 50) else 'chronic pain'), axis=1)

    def exclude_pain(self, threshold: int = 20) -> None:
        
        """
        Function to recode the pain scores into a binary variable

        Parameters
        ----------
        threshold : int
            The threshold for dissociating pain groups

        Returns (Internal)
        ------------------
        self.data : pd.DataFrame
            The data with the depression scores recoded
        """
        
        #Determine number of participants
        num_participants = self.data['participant_id'].nunique()

        #Remove the participants who are in the transition group (duration = 3 – 6 months)
        nontransition_index = self.data['duration'] == '3 – 6 months'
        self.data.loc[nontransition_index, self.group_code] = 'acute pain'

        #Create composite score
        self.data['composite_pain'] = self.data.apply(lambda x: np.mean((x['intensity'], x['unpleasant'], x['interference'])), axis=1)

        #Find row indexes of participants in no pain (column 'group_code') where their average_pain scores are less than the threshold
        no_pain_participants = (self.data[self.group_code] == 'no pain') & (self.data['composite_pain'] < threshold)
        acute_pain_participants = (self.data[self.group_code] == 'acute pain') & (self.data['composite_pain'] >= threshold)
        chronic_pain_participants = (self.data[self.group_code] == 'chronic pain') & (self.data['composite_pain'] >= threshold)
        kept_participants = no_pain_participants | acute_pain_participants | chronic_pain_participants

        self.data = self.data[kept_participants].reset_index(drop=True)
        self.excluded_pain = num_participants - self.data['participant_id'].nunique()

    def save_processed_data(self) -> None:

        """
        Function to save the processed data to a new file

        Returns (External)
        ------------------
        Data: csv
            The processed data
            The learning data
            The transfer data
        """

        #Save the processed data to a new file
        self.learning_data.to_csv(os.path.join('SOMA_AL','data',f'{self.split_by_group}_learning_processed.csv'), index=False)
        self.transfer_data.to_csv(os.path.join('SOMA_AL','data',f'{self.split_by_group}_transfer_processed.csv'), index=False)

    #Data filtering
    def filter_learning_data(self) -> None:

        """
        Function to filter and manipulate the learning data

        Returns (Internal)
        ------------------
        self.learning_data : pd.DataFrame
            The filtered learning data
        """

        #Filter data
        self.learning_data = self.data[self.data['trial_type'] == 'learning-trials'].reset_index(drop=True)

        #Create symbol coding
        symbol_renames = {
            '75R1': 'Reward1', 
            '25R1': 'Reward1',
            '75R2': 'Reward2',
            '25R2': 'Reward2',
            '75P1': 'Punish1',
            '25P1': 'Punish1',
            '75P2': 'Punish2',
            '25P2': 'Punish2',
        }
        self.learning_data['symbol_names'] = self.learning_data['symbol_L_name']
        for symbol in symbol_renames:
            self.learning_data['symbol_names'] = self.learning_data['symbol_names'].replace(symbol, symbol_renames[symbol])
        
        self.learning_data['symbol_name'] = self.learning_data['symbol_names']
        self.learning_data['symbol_name'] = self.learning_data['symbol_name'].replace({'Reward1': 'Reward',
                                                                                        'Reward2': 'Reward', 
                                                                                        'Punish1': 'Punish',
                                                                                        'Punish2': 'Punish'})

        #Create trial indices - computes both cases (75R1, 75R2) of reward and both cases of punish (75P1, 75P2) seperately to later be averaged
        self.learning_data['trial_number'] = self.learning_data.groupby(['participant_id', 'symbol_names']).cumcount() + 1

        #Create binned trial indices
        self.learning_data['binned_trial'] = np.ceil(self.learning_data['trial_number'] / 6).astype(int)
        self.learning_data['binned_trial'].replace({1: 'Early', 2: 'Mid-Early', 3: 'Mid-Late', 4: 'Late'}, inplace=True)

        #Create trial indices per participant and symbol_name #TODO: Check this
        self.learning_data['trial_number_symbol'] = self.learning_data.groupby(['participant_id', 'symbol_names']).cumcount() + 1

    def filter_transfer_data(self) -> None:

        """
        Function to filter and manipulate the transfer data

        Returns (Internal)
        ------------------
        self.transfer_data : pd.DataFrame
            The filtered transfer data
        """

        #Filter data
        self.transfer_data = self.data[self.data['trial_type'] == 'probe'].reset_index(drop=True)
        self.transfer_data = self.transfer_data[self.transfer_data['symbol_L_value'] != self.transfer_data['symbol_R_value']] #Remove trials where the same valued symbol was presented together

        #Determine which symbol_n_value was chosen using the choice_made column where 1 = Right, 0 = Left in the transfer data
        self.transfer_data['symbol_chosen'] = self.transfer_data['symbol_L_value'] #Default to chose left
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_chosen'] = self.transfer_data['symbol_R_value'] #Switch to chose right if appropriate
        self.transfer_data['symbol_ignored'] = self.transfer_data['symbol_R_value']
        self.transfer_data.loc[self.transfer_data['choice_made'] == 1, 'symbol_ignored'] = self.transfer_data['symbol_L_value']

        #TODO: It keeps the novel stimuli
        high_reward = ((self.transfer_data['symbol_L_value'] == 4) & (self.transfer_data['symbol_R_value'] != 1)) | ((self.transfer_data['symbol_L_value'] != 1) & (self.transfer_data['symbol_R_value'] == 4))
        high_punish = ((self.transfer_data['symbol_L_value'] == 1) & (self.transfer_data['symbol_R_value'] != 4)) | ((self.transfer_data['symbol_L_value'] != 4) & (self.transfer_data['symbol_R_value'] == 1))
        moderate = ((self.transfer_data['symbol_L_value'] == 2) | (self.transfer_data['symbol_L_value'] == 3) | (self.transfer_data['symbol_L_value'] == 0)) & ((self.transfer_data['symbol_R_value'] == 2) | (self.transfer_data['symbol_R_value'] == 3) | (self.transfer_data['symbol_R_value'] == 0))
        self.transfer_data['context'] = np.nan
        self.transfer_data.loc[high_reward, 'context'] = 'high_reward'
        self.transfer_data.loc[high_punish, 'context'] = 'high_punish'
        self.transfer_data.loc[moderate, 'context'] = 'moderate'

        self.transfer_data['paired_symbols'] = self.transfer_data.apply(self.combine_columns, axis=1)
    
        self.transfer_data = self.transfer_data[self.transfer_data['context_val'] == -1]
        #self.transfer_data = self.transfer_data[self.transfer_data['context_val'] == 1]
        #self.transfer_data = self.transfer_data[self.transfer_data['context_val'] == 0]
        
    #Data exclusion
    def exclude_low_accuracy(self, threshold: int = 55) -> None:

        """
        Function to exclude participants with low accuracy.

        Parameters
        -----------
        threshold : int
            The threshold for accuracy below which participants are excluded.

        Returns (Internal)
        ------------------
        self.data : pd.DataFrame
            The data with participants excluded
        self.learning_data : pd.DataFrame
            The learning data with participants excluded
        self.transfer_data : pd.DataFrame
            The transfer data with participants excluded
        self.participants_excluded_accuracy : int
            The number of participants excluded
        self.accuracy_threshold : int
            The threshold for accuracy below which participants are excluded
        """

        #Compute accuracy for each participant
        accuracy = pd.DataFrame(columns=['accuracy'], index=pd.MultiIndex(levels=[[]], codes=[[]], names=['participant']))
        for participant in self.learning_data['participant_id'].unique():
            participant_data = self.learning_data[self.learning_data['participant_id'] == participant]
            accuracy_rate = participant_data['accuracy'].mean()
            accuracy.loc[participant, 'accuracy'] = accuracy_rate

        #Find participants with accuracy less than 60%
        low_accuracy = accuracy[accuracy['accuracy'] < threshold].reset_index()
        excluded_participants_learning = low_accuracy['participant']

        choice_rate = self.choice_rate.reset_index()
        choice_rate = choice_rate[(choice_rate['symbol'] == 'High Reward') | (choice_rate['symbol'] == 'High Punish')]
        reduced_choice_rate = pd.DataFrame(columns=['participant_id', 'High Reward', 'High Punish'])
        for participant in choice_rate['participant_id'].unique():
            participant_choice_rate = choice_rate[choice_rate['participant_id'] == participant]
            participant_choice_rate = participant_choice_rate.pivot(index='participant_id', columns='symbol', values='choice_rate').reset_index()
            participant_choice_rate['High Punish'] = 100-participant_choice_rate['High Punish']
            reduced_choice_rate = pd.concat((reduced_choice_rate, participant_choice_rate))
        #Find whether any High Reward or High Punish is < threshold
        low_choice_rate = reduced_choice_rate[(reduced_choice_rate['High Reward'] < threshold) | (reduced_choice_rate['High Punish'] < threshold)]
        excluded_participants_transfer = low_choice_rate['participant_id']

        excluded_participants = pd.concat([excluded_participants_learning, excluded_participants_transfer]).drop_duplicates()

        #Remove participants with accuracy less than 60%
        self.data = self.data[~self.data['participant_id'].isin(excluded_participants)]
        self.learning_data = self.learning_data[~self.learning_data['participant_id'].isin(excluded_participants)]
        self.transfer_data = self.transfer_data[~self.transfer_data['participant_id'].isin(excluded_participants)]

        #Track number of participants excluded
        self.participants_excluded_accuracy = len(excluded_participants)
        self.accuracy_threshold = threshold

    def exclude_low_rt(self, low_threshold: int = 200, high_threshold: int = 5000) -> None:

        """
        Function to exclude trials with low/high reaction times.

        Parameters
        ----------
        low_threshold : int
            The lower threshold for reaction time below which trials are excluded.
        high_threshold : int
            The upper threshold for reaction time above which trials are excluded.

        Returns (Internal)
        ------------------
        self.learning_data : pd.DataFrame
            The learning data with trials excluded
        self.transfer_data : pd.DataFrame
            The transfer data with trials excluded
        self.trials_excluded_rt : float
            The percentage of trials excluded
        """

        self.learning_data['excluded_rt'] = (self.learning_data['rt'] < low_threshold) | (self.learning_data['rt'] > high_threshold)
        learning_excluded, learning_trials = self.learning_data['excluded_rt'].sum(), self.learning_data.shape[0]
        self.learning_data = self.learning_data[self.learning_data['excluded_rt'] == False]

        self.transfer_data['excluded_rt'] = (self.transfer_data['rt'] < low_threshold) | (self.transfer_data['rt'] > high_threshold)
        transfer_excluded, transfer_trials = self.transfer_data['excluded_rt'].sum(), self.transfer_data.shape[0]
        self.transfer_data = self.transfer_data[self.transfer_data['excluded_rt'] == False]

        #Track number of participants excluded
        self.trials_excluded_rt = (learning_excluded + transfer_excluded)/(learning_trials + transfer_trials) * 100

    #Data collapsing and transformation
    def average_data(self) -> None:

        """
        Function to average the data for each participant and symbol

        Returns (Internal)
        ------------------
        self.avg_learning_data : pd.DataFrame
            The averaged learning data
        self.avg_transfer_data : pd.DataFrame
            The averaged transfer data
        """

        #Create participant average for learning data
        self.avg_learning_data = self.learning_data.groupby(['participant_id', self.group_code, 'symbol_name'])['accuracy'].mean().reset_index()
        self.avg_learning_data['context'] = -1
        self.avg_learning_data.loc[self.avg_learning_data['symbol_name'] == 'Reward', 'context'] = 1
        if self.split_by_group == 'pain':
            self.avg_learning_data['group'] = 0
            self.avg_learning_data.loc[self.avg_learning_data[self.group_code] == 'no pain', 'group'] = -1
            self.avg_learning_data.loc[self.avg_learning_data[self.group_code] == 'chronic pain', 'group'] = 1
        else:
            self.avg_learning_data['group'] = -1
            self.avg_learning_data.loc[self.avg_learning_data[self.group_code] == 'depressed', 'group'] = 1

        #Create participant average for transfer data
        self.transfer_data_reduced = self.transfer_data[~self.transfer_data['paired_symbols'].isna()]

        self.avg_transfer_data = self.transfer_data.groupby(['participant_id', self.group_code, 'paired_symbols'])['accuracy'].mean().reset_index()
        if self.split_by_group == 'pain':
            self.avg_transfer_data['group'] = 0
            self.avg_transfer_data.loc[self.avg_transfer_data[self.group_code] == 'no pain', 'group'] = -1
            self.avg_transfer_data.loc[self.avg_transfer_data[self.group_code] == 'chronic pain', 'group'] = 1
        else:
            self.avg_transfer_data['group'] = -1
            self.avg_transfer_data.loc[self.avg_transfer_data[self.group_code] == 'depressed', 'group'] = 1

        #Save to csv
        self.avg_learning_data.to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_learning_data.csv', index=False)
        self.learning_data.to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_learning_data_trials.csv', index=False)

        #Create participant average for transfer data
        self.avg_transfer_data.to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_transfer_data.csv', index=False)
        self.transfer_data.to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_transfer_data_trials.csv', index=False)
        self.transfer_data_reduced.to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_transfer_data_trials_reduced.csv', index=False)

    def average_transform_data(self, data: pd.DataFrame, metric: str, factor: str, transformation: str) -> pd.DataFrame:

        """
        Function to transform and average the data

        Parameters
        ----------
        data : pd.DataFrame
            The data to transform and average
        metric : str
            The metric to transform and average
        factor : str
            The factor to average the data by
        transformation : str
            The transformation to apply to the metric

        Returns
        -------
        data : pd.DataFrame
            The transformed and averaged data
        """

        data = self.transform_data(data, metric, transformation)
        data = self.average_byfactor(data, metric, factor)

        return data
    
    def average_byfactor(self, data: pd.DataFrame, metric: str, factor: str) -> pd.DataFrame:

        """
        Function to average the data relative to a factor

        Parameters
        ----------
        data : pd.DataFrame
            The data to average
        metric : str
            The metric to average
        factor : str
            The factor to average the data by

        Returns
        -------
        data : pd.DataFrame
            The averaged data
        """

        avg_factor = [factor] if type(factor) == str else factor

        return data.groupby(['participant_id']+avg_factor)[metric].mean().reset_index()  
    
    def transform_data(self, data: pd.DataFrame, metric: str, transformation: str) -> pd.DataFrame:

        """
        Function to transform the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data to transform
        metric : str
            The metric to transform
        transformation : str
            The transformation to apply to the metric

        Returns
        -------
        data : pd.DataFrame
            The transformed data
        """

        data[metric] = data[metric].transform(lambda x: eval(transformation))

        return data
    
    def manipulate_data(self, data: pd.DataFrame, metric: str, factor: str, transformation: str) -> pd.DataFrame:

        """
        Function to manipulate the data based on a transformation

        Parameters
        ----------
        data : pd.DataFrame
            The data to manipulate
        metric : str
            The metric to manipulate
        factor : str
            The factor to manipulate the data by
        transformation : str
            The transformation to apply to the data

        Returns
        -------
        manipulated_data : pd.DataFrame
            The manipulated data
        """

        conditions = transformation.split('-')

        if '-' in transformation:
            manipulated_data = data[(data[factor] == conditions[0]) | (data[factor] == conditions[1])]
            manipulated_data = manipulated_data.sort_values(by=['participant_id', factor])
            manipulated_data = manipulated_data.groupby('participant_id')[metric].diff()
            manipulated_data = pd.concat([data[[self.group_code, factor, 'participant_id']], manipulated_data], axis=1).dropna()
            manipulated_data[factor] = transformation
        else:
            warnings.warn('Transformation not recognized, only subtraction currently implemented. Returning original data.')
            manipulated_data = data

        return manipulated_data
    
    #Metric computations
    def compute_accuracy(self) -> None:

        """
        Function to compute the accuracy of the learning and transfer data

        Returns (Internal)
        ------------------
        self.learning_data : pd.DataFrame
            The learning data with accuracy computed
        self.transfer_data : pd.DataFrame
            The transfer data with accuracy computed
        """
        
        #Compute learning accuracy
        self.learning_data['larger_value'] = (self.learning_data['symbol_R_value'] > self.learning_data['symbol_L_value']).astype(int) #1 = Right has larger value, 0 = Left has larger value
        self.learning_data['accuracy'] = (self.learning_data['larger_value'] == self.learning_data['choice_made']).astype(int)*100 #100 = Correct, 0 = Incorrect

        #Compute transfer accuracy
        self.transfer_data['larger_value'] = (self.transfer_data['symbol_R_value'] > self.transfer_data['symbol_L_value']).astype(int) #1 = Right has larger value, 0 = Left has larger value
        self.transfer_data['accuracy'] = (self.transfer_data['larger_value'] == self.transfer_data['choice_made']).astype(int)*100 #100 = Correct, 0 = Incorrect
        
    def compute_learning_averages(self) -> None:

        """
        Function to compute the learning averages

        Returns (Internal)
        ------------------
        self.learning_accuracy : pd.DataFrame
            The averaged learning accuracy
        self.learning_accuracy_diff : pd.DataFrame
            The difference in learning accuracy
        self.learning_rt : pd.DataFrame
            The averaged learning reaction time
        self.learning_rt_diff : pd.DataFrame
            The difference in learning reaction time

        """
        
        #Compute accuracy for each participant and symbol_name within each group
        self.learning_accuracy = self.learning_data.groupby([self.group_code, 'participant_id', 'symbol_name'])['accuracy'].mean().reset_index()
        self.learning_accuracy['symbol_name'] = pd.Categorical(self.learning_accuracy['symbol_name'], ['Reward', 'Punish'])
        self.learning_accuracy = self.learning_accuracy.sort_values(by=['participant_id', 'symbol_name'])

        self.learning_accuracy_diff = self.learning_accuracy.groupby(['participant_id'])['accuracy'].diff()
        self.learning_accuracy_diff = pd.concat([self.learning_accuracy[[self.group_code, 'participant_id']], self.learning_accuracy_diff], axis=1).dropna()
        self.learning_accuracy_diff['symbol_name'] = 'Difference' 

        self.learning_rt = self.learning_data.groupby([self.group_code, 'participant_id', 'symbol_name'])['rt'].mean().reset_index()
        self.learning_rt['symbol_name'] = pd.Categorical(self.learning_rt['symbol_name'], ['Reward', 'Punish'])
        self.learning_rt = self.learning_rt.sort_values(by=['participant_id', 'symbol_name'])

        self.learning_rt_diff = self.learning_rt.groupby(['participant_id'])['rt'].diff()
        self.learning_rt_diff = pd.concat([self.learning_rt[[self.group_code, 'participant_id']], self.learning_rt_diff], axis=1).dropna()
        self.learning_rt_diff['symbol_name'] = 'Difference'

        self.learning_accuracy.set_index([self.group_code, 'participant_id', 'symbol_name'], inplace=True)
        self.learning_accuracy_diff.set_index([self.group_code, 'participant_id', 'symbol_name'], inplace=True)
        self.learning_rt.set_index([self.group_code, 'participant_id', 'symbol_name'], inplace=True)
        self.learning_rt_diff.set_index([self.group_code, 'participant_id', 'symbol_name'], inplace=True)

    def compute_valence_bias(self) -> None:

        """
        Function to compute the valence bias

        Returns (Internal)
        ------------------
        self.valence_bias : pd.DataFrame
            The valence bias
        """

        #Compute valence bias
        choice_rate = self.choice_rate.reset_index()
        HR = choice_rate[choice_rate['symbol']=='High Reward']['choice_rate'].reset_index(drop=True)
        LR = choice_rate[choice_rate['symbol']=='Low Reward']['choice_rate'].reset_index(drop=True)
        LP = choice_rate[choice_rate['symbol']=='Low Punish']['choice_rate'].reset_index(drop=True)
        HP = choice_rate[choice_rate['symbol']=='High Punish']['choice_rate'].reset_index(drop=True)

        HRLR = HR - LR
        LPHP = LP - HP
        valence_bias = HRLR - LPHP

        #Create new valence bias dataframe
        self.valence_bias = choice_rate[choice_rate['symbol']=='High Reward'][[self.group_code, 'participant_id']].reset_index(drop=True)
        self.valence_bias['symbol'] = 'bias'
        self.valence_bias['valence_bias'] = pd.Series(valence_bias, dtype='float64')
        self.valence_bias = self.valence_bias.set_index([self.group_code, 'participant_id', 'symbol'])

        #Save to csv
        self.valence_bias.reset_index().to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_transfer_valence_bias.csv', index=False)

    def compute_choice_rate(self, neutral: bool = False) -> None:

        '''
        Function to compute the choice rate for each participant and symbol within each group

        parameters:
        -----------
        neutral : bool
            Whether to compute the choice rate for the neutral analyses

        Returns (Internal)
        ------------------
        self.choice_rate : pd.DataFrame
            The choice rate for each participant and symbol within each group
        self.choice_rt : pd.DataFrame
            The reaction time for each participant and symbol within each group
        self.neutral_choice_rate : pd.DataFrame
            The choice rate for each participant and symbol within each group for the neutral analyses
        self.neutral_choice_rt : pd.DataFrame
            The reaction time for each participant and symbol within each group for the neutral analyses
        '''

        data = self.transfer_data if not neutral else self.transfer_data[self.transfer_data['neutral_values']]

        #Compute choice rates for each participant and symbol within each group
        choice_rate = pd.DataFrame(columns=['choice_rate'], index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=[self.group_code, 'participant_id', 'symbol']))
        choice_rt = pd.DataFrame(columns=['choice_rt'], index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=[self.group_code, 'participant_id', 'symbol']))
        choice_rate_context = pd.DataFrame(columns=['choice_rate'], index=pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []], names=[self.group_code, 'participant_id', 'symbol', 'context_val']))

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
                    choice_rt.loc[(group, participant, symbol), 'choice_rt'] = participant_data[participant_data['symbol_chosen'] == symbol]['rt'].mean()

        for group in self.group_labels:
            group_data = data[data[self.group_code] == group]
            for participant in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant]
                symbols = [0, 1, 2, 3, 4] if not neutral else [2, 3]
                participant_data['context_val'] = participant_data['context_val'].replace({1: 'Reward', -1: 'Punish', 0: 'Neutral'})
                for context in participant_data['context_val'].unique():
                    context_data = participant_data[participant_data['context_val'] == context]
                    for symbol in symbols:
                        symbol_chosen = context_data[context_data['symbol_chosen'] == symbol].shape[0]
                        symbol_ignored = context_data[context_data['symbol_ignored'] == symbol].shape[0]
                        symbol_choice_rate = symbol_chosen / (symbol_chosen + symbol_ignored) * 100
                        choice_rate_context.loc[(group, participant, symbol, context), 'choice_rate'] = symbol_choice_rate

        if not neutral:
            choice_rate = choice_rate.reset_index()
            choice_rate['symbol'] = choice_rate['symbol'].replace({0: 'Novel', 1: 'High Punish', 2: 'Low Punish', 3: 'Low Reward', 4: 'High Reward'})
            choice_rate = choice_rate.set_index([self.group_code, 'participant_id', 'symbol'])

            choice_rate_context = choice_rate_context.reset_index()
            choice_rate_context['symbol'] = choice_rate_context['symbol'].replace({0: 'Novel', 1: 'High Punish', 2: 'Low Punish', 3: 'Low Reward', 4: 'High Reward'})
            choice_rate_context = choice_rate_context.set_index([self.group_code, 'participant_id', 'symbol', 'context_val'])

            choice_rt = choice_rt.reset_index()
            choice_rt['symbol'] = choice_rt['symbol'].replace({0: 'Novel', 1: 'High Punish', 2: 'Low Punish', 3: 'Low Reward', 4: 'High Reward'})
            choice_rt = choice_rt.set_index([self.group_code, 'participant_id', 'symbol'])

            self.choice_rate = choice_rate
            self.choice_rate_context = choice_rate_context
            self.choice_rt = choice_rt

            self.choice_rate.reset_index().to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_choice_rates.csv', index=False)
            self.choice_rate_context.reset_index().to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_choice_rates_context.csv', index=False)
            self.choice_rt.reset_index().to_csv(f'SOMA_AL/stats/{self.split_by_group}_stats_choice_rt.csv', index=False)

        else:
            choice_rate = choice_rate.reset_index()
            choice_rate = choice_rate[choice_rate['symbol'] == 3] #Choose rewarding 
            choice_rate = choice_rate.set_index([self.group_code, 'participant_id', 'symbol'])
            self.neutral_choice_rate = choice_rate

            choice_rt = choice_rt.reset_index()
            choice_rt = choice_rt[choice_rt['symbol'] == 3] #Choose rewarding
            choice_rt = choice_rt.set_index([self.group_code, 'participant_id', 'symbol'])
            self.neutral_choice_rt = choice_rt

    def compute_demographics(self) -> None:

        """
        Function to compute the demographics of the participants

        Returns (Internal)
        ------------------
        self.demographics : pd.DataFrame
            The demographics of the participants
        self.mean_age : pd.Series
            The mean age of the participants
        self.std_age : pd.Series
            The standard deviation of the age of the participants
        self.female_counts : pd.Series
            Count of number of females
        self.male_counts : pd.Series
            Count of number of males
        self.not_specified_counts : pd.Series
            Count of number of participants who did not specify
        self.demo_sample_size : pd.Series
            The sample size for each group
        self.demo_age : pd.Series
            The mean age of the participants
        self.demo_gender : pd.Series
            Counts of gender
        self.demographics_summary : pd.DataFrame
            The demographics summary
        """

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
    def compute_pain_scores(self) -> None:

        """
        Function to compute the pain scores

        Returns (Internal)
        ------------------
        self.pain_scores : pd.DataFrame
            The pain scores
        self.mean_pain : pd.DataFrame
            The mean pain scores
        self.std_pain : pd.DataFrame
            The standard deviation of the pain scores
        self.pain_summary : pd.DataFrame
            The pain summary
        """

        self.pain_scores = self.data.groupby([self.group_code, 'participant_id'])[['intensity', 'unpleasant', 'interference']].first().reset_index()
        self.mean_pain = self.data.groupby(self.group_code)[['intensity', 'unpleasant', 'interference']].mean()
        self.std_pain = self.data.groupby(self.group_code)[['intensity', 'unpleasant', 'interference']].std()
        self.pain_summary = self.mean_pain.round(2).astype(str) + ' (' + self.std_pain.round(2).astype(str) + ')'
        self.pain_summary = self.pain_summary.reindex(self.group_labels)
        self.pain_summary = self.pain_summary.T

    #Compute depression scores
    def compute_depression_scores(self) -> None:

        """
        Function to compute the depression scores

        Returns (Internal)
        ------------------
        self.depression_scores : pd.DataFrame
            The depression scores
        self.mean_depression : pd.DataFrame
            The mean depression scores
        self.std_depression : pd.DataFrame
            The standard deviation of the depression scores
        self.depression_summary : pd.DataFrame
            The depression summary
        """
        
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