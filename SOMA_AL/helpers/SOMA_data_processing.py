#System settings
import sys
sys.dont_write_bytecode = True

#Import modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.print_filename = f'{self.print_filename.replace(".txt","")}_{file_name.replace(".csv","")}.txt'

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
    
    def print_report(self):

        #Initiate the report
        with open(self.print_filename, 'w') as f:
            f.write(f'SOMA Report\nFile: {self.file_name}\n')
            f.write('\n')

        #Populate the report
        self.print_data('Column Names:', ', '.join(self.data.columns))
        self.print_data('Data Dimensions:', 
                        [['Rows', 'Columns', 'Number of Participants'],
                         [self.data.shape[0], self.data.shape[1], self.data['participant_id'].nunique()]])
        self.print_data('Data Head:', self.data.head())
        #self.print_data('Data Summary:', self.data_summary) #TODO: Removed because it collapses across all conditions and groups
        self.print_data('Grouped Summary of Pain:', self.grouped_summary)

    #### PLOTS ####
    def print_plots(self):
        self.plot_learning_accuracy(rolling_mean=True)
        self.plot_transfer_accuracy()

    def plot_learning_accuracy(self, rolling_mean=False, CIs=False):
        #Add three sublpots, one for each group (group_code), which shows the average accuracy over trials (trial_number) for each of the two contexts (context_val_name)
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_data = self.learning_data[self.learning_data['group_code'] == group]
            for context_index, context in enumerate(['Reward', 'Loss Avoid']):
                context_data = group_data[group_data['context_val_name'] == context]
                mean_accuracy = context_data.groupby('trial_number')['accuracy'].mean()*100
                if rolling_mean:
                    mean_accuracy = mean_accuracy.rolling(5).mean()
                std_accuracy = context_data.groupby('trial_number')['accuracy'].std()
                if CIs:
                    ax[i].fill_between(mean_accuracy.index, mean_accuracy - 1.96*std_accuracy, mean_accuracy + 1.96*std_accuracy, alpha=0.2, color=['C0', 'C1'][context_index])
                ax[i].plot(mean_accuracy, label=context, color=['C0', 'C1'][context_index])

            ax[i].set_ylim(50, 90)
            ax[i].set_title(f'{group.capitalize()}')
            ax[i].set_xlabel('Trial Number')
            ax[i].set_ylabel('Accuracy')
            ax[i].legend()

        #Add a fourth subplot to show the difference in accuracy between the two contexts for each group across trials
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_data = self.learning_data[self.learning_data['group_code'] == group]
            context_data = group_data.groupby(['trial_number', 'context_val_name'])['accuracy'].mean().unstack()
            context_data['Difference'] = context_data['Loss Avoid'] - context_data['Reward']
            ax[3].plot(context_data['Difference'], label=group)

        ax[3].set_title('Difference in Accuracy')
        ax[3].set_xlabel('Trial Number')
        ax[3].set_ylabel('Accuracy Difference')
        ax[3].legend()
        
        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure 2A - Accuracy Across Learning.png')

        #Close figure
        plt.close()

    def plot_transfer_accuracy(self):
        #Determine how often each symbol was chosen in the transfr trials and divide by the number of times it was shown to get the probability of choosing each symbol but split per group per participant_id within each group
        symbol_chosen_counts = self.transfer_data.groupby(['group_code', 'symbol_chosen', 'participant_id'])['symbol_chosen'].count()
        symbol_shown_counts = self.transfer_data.groupby(['group_code', 'symbol_chosen', 'participant_id'])['symbol_chosen'].count() + self.transfer_data.groupby(['group_code', 'symbol_ignored', 'participant_id'])['symbol_ignored'].count()
        choice_rate = (symbol_chosen_counts / symbol_shown_counts) * 100

        #Create a bar plot of the choice rate for each symbol
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_choice_rate = choice_rate.loc[group]
            #create a boxplot of the choice rate for each symbol
            bplot = ax[i].boxplot([group_choice_rate.loc[0], group_choice_rate.loc[1], group_choice_rate.loc[2], group_choice_rate.loc[3], group_choice_rate.loc[4]], 
                                  patch_artist=True, meanline=True, showmeans=True, showfliers=False,
                                  tick_labels=['Novel', 'High\nPunish', 'Low\nPunish', 'Low\nReward', 'High\nReward'])            
            colors = ['#D3D3D3', '#FF0000', '#FFB6C1', '#90EE90', '#00FF00']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)            
            ax[i].invert_xaxis()
            ax[i].set_xlabel('')
            ax[i].set_ylabel('Choice Rate (%)')
            ax[i].set_ylim(0, 90)
            ax[i].set_title(group.capitalize())

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure 2B - Transfer Choice Rate.png')

        #Close figure
        plt.close()

    #### TESTS ####
    def run_tests(self):
        self.test_trial_counts()

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

    #### DESCRIPTIVES ####
    def compute_learning_accuracy(self):
        pass
        #GROUP x REWARD/PUNISH across trials
        #GrouP learning data by group_code and context_val_name

def main():
    #Initiate pipeline
    SOMA_pipeline = SOMAALPipeline()

    #Load data
    SOMA_pipeline.load_data(file_path=r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific', 
                    file_name= [r'v1a_avoid_pain\v1a_avoid_pain.csv',r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv'])
    
    #Process data
    SOMA_pipeline.process_data()
    #SOMA_pipeline.save_processed_data()

    #Compute summary statistics
    SOMA_pipeline.summarize_data()
    SOMA_pipeline.groupby_summary('group_code')

    #Test code
    SOMA_pipeline.run_tests()

    #Report data
    SOMA_pipeline.print_report()
    SOMA_pipeline.print_plots()

    #Compute descriptives
    SOMA_pipeline.compute_learning_accuracy()

    #Debug tag
    print()

if __name__ == '__main__':
    main()
    print()


    