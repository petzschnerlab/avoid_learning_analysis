import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SOMATests:

    """
    Class to run tests for the SOMA project
    """

    def run_tests(self):
        self.test_trial_counts()
        if self.tests == 'extensive':
            self.test_plot_learning_accuracy(self.test_rolling_mean)
        self.test_determine_contingencies()

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
            
    def test_plot_learning_accuracy(self, rolling_mean=None):

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
            for participant in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant]
                for context_index, context in enumerate(['Reward', 'Loss Avoid']):
                    context_data = participant_data[participant_data['context_val_name'] == context]['accuracy']
                    if rolling_mean is not None:
                        context_data = context_data.rolling(rolling_mean, min_periods=1).mean()
                    plt.plot(np.arange(1,context_data.shape[0]+1) ,context_data, color=['#B2DF8A', '#FB9A99'][context_index], label=['Reward' if context == 'Reward' else 'Punish'])

                plt.ylim(-5, 105)
                plt.title(f'{participant.capitalize()}')
                plt.xlabel('Trial Number')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right', frameon=False)

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
            if abs(feedback_freqs[symbol][0] - feedback_freqs[symbol][1]) > 5:
                raise ValueError(f'Feedback frequencies for symbol {symbol} are incorrect: {feedback_freqs[symbol]}')


            

            
