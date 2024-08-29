import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Tests:

    """
    Class to run tests for the SOMA project
    """

    def run_tests(self):
        self.test_trial_counts()
        if self.tests == 'extensive':
            self.test_plot_learning_accuracy(self.test_rolling_mean, self.test_context_type)
        self.test_determine_contingencies()
        self.check_condition_order()

    def test_trial_counts(self):

        #for each articiant plot the trial indexes as the y axis using matlpotlib
        for participant in self.learning_data['participant_id'].unique():
            participant_data = self.learning_data[self.learning_data['participant_id'] == participant]
            participant_trial_counts = []
            for context in participant_data['context_val_name'].unique():
                context_data = participant_data[participant_data['context_val_name'] == context]
                participant_trial_counts.append(context_data.shape[0])

            if participant_trial_counts[0] != 48 or participant_trial_counts[1] != 48:
                raise ValueError(f'Participant {participant} has incorrect number of trials: {participant_trial_counts}') 
            
    def test_plot_learning_accuracy(self, rolling_mean=None, context_type='context'):

        #Create folder for plots if non existent
        if not os.path.exists('SOMA_AL/plots/tests'):
            os.makedirs('SOMA_AL/plots/tests')
        if not os.path.exists('SOMA_AL/plots/tests/no pain'):
            os.makedirs('SOMA_AL/plots/tests/no pain')
        if not os.path.exists('SOMA_AL/plots/tests/acute pain'):
            os.makedirs('SOMA_AL/plots/tests/acute pain')
        if not os.path.exists('SOMA_AL/plots/tests/chronic pain'):
            os.makedirs('SOMA_AL/plots/tests/chronic pain')
        if not os.path.exists('SOMA_AL/plots/tests/healthy'):
            os.makedirs('SOMA_AL/plots/tests/healthy')
        if not os.path.exists('SOMA_AL/plots/tests/depressed'):
            os.makedirs('SOMA_AL/plots/tests/depressed')

        for i, group in enumerate(self.group_labels):
            folder_name = f'SOMA_AL/plots/tests/{group}/'
            group_data = self.learning_data[self.learning_data[self.group_code] == group]
            
            if context_type == 'context':
                group_data.loc[:,'symbol_names'] = group_data['symbol_names'].replace({'Reward1': 'Reward',
                                                                                        'Reward2': 'Reward', 
                                                                                        'Punish1': 'Punish',
                                                                                        'Punish2': 'Punish'})
                #Average duplicate trial_numbers for each participant within each symbol_name
                group_data = group_data[['participant_id', 'trial_number', 'symbol_names', 'accuracy']]
                group_data = group_data.groupby(['participant_id', 'trial_number', 'symbol_names']).mean().reset_index()
                contexts = ['Reward', 'Punish']
            else:
                contexts = ['Reward1', 'Reward2']#, 'Punish1', 'Punish2']

            color = ['#B2DF8A', '#FB9A99'] if context_type == 'context' else ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C']
            for participant in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant]
                for context_index, context in enumerate(contexts):
                    context_data = participant_data[participant_data['symbol_names'] == context]['accuracy']
                    if rolling_mean is not None:
                        context_data = context_data.rolling(rolling_mean, min_periods=1).mean()
                    plt.scatter(np.arange(1,context_data.shape[0]+1) ,context_data, color=color[context_index], label=context)

                plt.ylim(-5, 105)
                plt.title(f'{participant.capitalize()}')
                plt.xlabel('Trial Number')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right', frameon=False)
                plt.axvline(x=15, color='black', linestyle='--')
                plt.axvline(x=5, color='black', linestyle='--')
                plt.axvline(x=10, color='black', linestyle='--')
                plt.axhline(y=50, color='black', linestyle='--')

                #Save the plot
                plt.savefig(f'{folder_name}{participant}.png')

                #Close figure
                plt.close()

    def test_determine_contingencies(self):

        #Determine the symbols from symbol_L_name and symbol_R_name for the first and second half of data based on trial_number
        split_symbols = []
        for half in range(2):

                #Get data for each half splitting by trial_number
                data_index = self.learning_data['trial_number']<int(self.learning_data['trial_number'].max()/2) if half == 0 else self.learning_data['trial_number']>=int(self.learning_data['trial_number'].max()/2)
                half_data = self.learning_data[data_index]
    
                #Get symbols for each half
                symbols_L = half_data['symbol_L_name'].unique()
                symbols_R = half_data['symbol_R_name'].unique()
                #combine arrays
                symbols = np.unique(np.concatenate([symbols_L, symbols_R]))
                split_symbols.append(symbols)

        #Assertions to ensure symbols are the same in each half
        if not np.array_equal(split_symbols[0], split_symbols[1]):
            raise ValueError(f'Symbols are not the same in each half of the data: {split_symbols[0]}, {split_symbols[1]}')

        #Split groups by symbol_L_value and symbol_R_value and determine counts and contingencies for each symbol
        feedback_counts = {}
        feedback_freqs = {}
        for symbol in self.learning_data['symbol_L_name'].unique():

            #Get data for each symbol
            symbol_data_L = self.learning_data[self.learning_data['symbol_L_name'] == symbol]
            symbol_data_R = self.learning_data[self.learning_data['symbol_R_name'] == symbol]

            #Determine unique feedbacks
            feedback = pd.concat([symbol_data_L['feedback_L'], symbol_data_R['feedback_R']]).unique()
            feedback_counts[symbol] = list(feedback)

            #Determine percentage of feedbacks that are not zero for the first and second half of the data
            feedback_freq = []
            for half in range(2):
                
                #Get data for each half splitting by trial_number
                symbol_data_L_index = symbol_data_L['trial_number']<int(symbol_data_L['trial_number'].max()/2) if half == 0 else symbol_data_L['trial_number']>=int(symbol_data_L['trial_number'].max()/2)
                symbol_data_R_index = symbol_data_R['trial_number']<int(symbol_data_R['trial_number'].max()/2) if half == 0 else symbol_data_R['trial_number']>=int(symbol_data_R['trial_number'].max()/2)
                symbol_data_L_half = symbol_data_L[symbol_data_L_index]
                symbol_data_R_half = symbol_data_R[symbol_data_R_index]

                #Get feedback counts that are zero
                feedback_data_L_half = symbol_data_L_half[symbol_data_L_half['feedback_L'] == 0]
                feedback_data_R_half = symbol_data_R_half[symbol_data_R_half['feedback_R'] == 0]

                #Determine percentage of feedbacks that are not zero
                feedback_count = (feedback_data_L_half.shape[0] + feedback_data_R_half.shape[0])/(symbol_data_L_half.shape[0] + symbol_data_R_half.shape[0])*100
                feedback_count = 100 - feedback_count if 'R' in symbol else feedback_count
                feedback_freq.append(int(np.round(feedback_count)))

            #Store feedback frequencies
            feedback_freqs[symbol] = feedback_freq

        #Assertions to ensure feedback counts only contain 2 values
        for symbol in feedback_counts:
            if len(feedback_counts[symbol]) != 2:
                raise ValueError(f'Feedback counts for symbol {symbol} are incorrect: {feedback_counts[symbol]}')
            
        #Assertions to ensure feedback frequencies are similar in each symbol
        for symbol in feedback_freqs:
            if abs(feedback_freqs[symbol][0] - feedback_freqs[symbol][1]) > 10:
                raise ValueError(f'Feedback frequencies for symbol {symbol} are incorrect: {feedback_freqs[symbol]}')
            
    def check_condition_order(self):

        #Check whether the conditions are in the same order for each participant using symbol_L_name
        participant_order = []
        for participant in self.learning_data['participant_id'].unique():
            participant_data = self.learning_data[self.learning_data['participant_id'] == participant]
            participant_order.append(participant_data['symbol_names'].values)
            
        #Assertions to ensure conditions are not in the same order for each participant
        for i in range(len(participant_order)-1):
            if np.array_equal(participant_order[i], participant_order[i+1]):
                raise ValueError(f'Conditions are in the same order for participant {i+1} and participant {i+2}')


            

            
