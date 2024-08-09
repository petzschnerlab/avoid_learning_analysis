#System settings
import sys
sys.dont_write_bytecode = True

#Import modules
import os
import subprocess
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from markdown_pdf import MarkdownPdf, Section

#SOMAALPipeline class
class SOMAALPipeline:
    #Create a constructor method
    def __init__(self, print_filename=r'SOMA_AL/reports/SOMA_report.pdf'):
        self.print_filename = print_filename

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
    
    #### PRINTING REPORTS ####

    def add_data_pdf(self, content:list, toc:bool=True, center:bool=False):
        #Formatting
        user_css = 'h1 {text-align:center;}' if center else None
        section = Section(' \n '.join(content), toc=toc)
        self.pdf.add_section(section, user_css=user_css)

    def table_to_pdf(self, table:pd.DataFrame, floatfmt=".2f"):
        table = table.transpose()
        table = table.reset_index(level=[0,1])
        for i in range(1, table.shape[0]):
            if table['level_0'][i] == table['level_0'][i-1]:
                table['level_0'][i] = ''
        for i in range(table.shape[0]):
            table['level_0'][i] = f'**{table["level_0"][i].capitalize()}**' if table['level_0'][i] != '' else ''
        for i in range(table.shape[0]):
            table['level_1'][i] = f'**{table["level_1"][i].capitalize()}**'
        table.columns = table.columns.str.title()
        table.columns.values[0] = ''
        table.columns.values[1] = ''

        return table.to_markdown(floatfmt=floatfmt, index=False)

    def save_report(self):
        try:
            #Save pdf with default filename
            self.pdf.save(self.print_filename)
        except: 
            #If file is opened, it will need to save with alternative filename
            original_filename = self.print_filename
            i = 1
            while os.path.exists(self.print_filename):
                try:
                    self.print_filename = original_filename.replace('.pdf', f'-{i}.pdf')
                    self.pdf.save(self.print_filename)
                    break
                except:
                    i += 1
                
            #Raise warning
            warnings.warn(f'File {original_filename} is currently opened. Saving as {self.print_filename}', stacklevel=2)
            
    def build_report(self):

        #Initiate the printing of plots
        self.print_plots()

        #Initiate report class
        self.pdf = MarkdownPdf(toc_level=3)

        #Add title Page
        section_text = [f'# SOMA Report',
                        f'![SOMA_logo](SOMA_AL/media/SOMA_preview.png)']
        self.add_data_pdf(section_text)

        #Add report details
        section_text = [f'## SOMA Report Details',
                        f'**Generated by:** Chad C. Williams\n',
                        f'**Date:** {pd.Timestamp.now()}']
        self.add_data_pdf(section_text)

        #Add data characteristics
        section_text = [f'## Data Characteristics',
                        f'**File{"s" if len(self.file_name) > 1 else ""}:** {", ".join(self.file_name)}',
                        f'### Column Names',
                        f'{", ".join(self.data.columns)}',
                        f'### Data Dimensions',
                        f'**Rows:** {self.data.shape[0]}\n',
                        f'**Columns:** {self.data.shape[1]}\n',
                        f'**Number of Participants:** {self.data["participant_id"].nunique()}']
        self.add_data_pdf(section_text)

        #Add data summary
        section_text = [f'## Participant Characteristics',
                        f'**Grouped Summary of Pain**',
                        #f'<center>',
                        f'{self.table_to_pdf(self.grouped_summary)}',
                        #f'</center>'
                        ]

        self.add_data_pdf(section_text, center=True)

        #Add behavioural findings
        figure_1_caption = '**Figure 1.** Behavioral performance across learning trials for the rich and poor contexts for each group.'
        if self.fig1_rolling_mean is not None:
            figure_1_caption += f' For visualization, the accuracy is smoothed using a rolling mean of {self.fig1_rolling_mean} trials.'

        if self.fig1_CIs is not False:
            figure_1_caption += ' Shaded regions represent 95\% confidence intervals.'

        section_text = [f'## Behavioural Findings',
                        f'### Learning Accuracy',
                        f'![learning_accuracy](SOMA_AL/plots/Figure_2A_Accuracy_Across_Learning.png)',
                        f'{figure_1_caption}',
                        f'### Transfer Accuracy',
                        f'![transfer_choice](SOMA_AL/plots/Figure_2B_Transfer_Choice_Rate.png)',
                        f"""**Figure 2.** Choice rate for each symbol during transfer trials for each group.
                        Choice rate is computed as the number of times a symbol was chosen given the number of times it was presented.
                        Boxlpots show the mean and standard deviation of the choice rate for each symbol type across participants within each group."""]
        self.add_data_pdf(section_text)

        #Save to pdf
        self.save_report()

    #### PLOTS ####
    def print_plots(self):
        self.plot_learning_accuracy(rolling_mean=3, CIs=True)
        self.plot_transfer_accuracy()

    def plot_learning_accuracy(self, rolling_mean=None, CIs=False):
        #Track the rolling mean
        self.fig1_rolling_mean = rolling_mean
        self.fig1_CIs = CIs

        #Add three sublpots, one for each group (group_code), which shows the average accuracy over trials (trial_number) for each of the two contexts (context_val_name)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_data = self.learning_data[self.learning_data['group_code'] == group]
            for context_index, context in enumerate(['Reward', 'Loss Avoid']):
                context_data = group_data[group_data['context_val_name'] == context]
                mean_accuracy = context_data.groupby('trial_number')['accuracy'].mean()*100
                if rolling_mean is not None:
                    mean_accuracy = mean_accuracy.rolling(rolling_mean).mean()
                std_accuracy = context_data.groupby('trial_number')['accuracy'].std()
                if CIs: #TODO: CHECK THESE
                    ax[i].fill_between(mean_accuracy.index, mean_accuracy - 1.96*std_accuracy, mean_accuracy + 1.96*std_accuracy, alpha=0.2, color=['C0', 'C1'][context_index])
                ax[i].plot(mean_accuracy, label=context, color=['C0', 'C1'][context_index])

            ax[i].set_ylim(40, 100)
            ax[i].set_title(f'{group.capitalize()}')
            ax[i].set_xlabel('Trial Number')
            ax[i].set_ylabel('Accuracy')
            ax[i].legend(loc='lower right', frameon=False)

        #Add a fourth subplot to show the difference in accuracy between the two contexts for each group across trials
        '''
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_data = self.learning_data[self.learning_data['group_code'] == group]
            context_data = group_data.groupby(['trial_number', 'context_val_name'])['accuracy'].mean().unstack()
            context_data['Difference'] = context_data['Loss Avoid'] - context_data['Reward']
            ax[3].plot(context_data['Difference'], label=group)

        ax[3].set_title('Difference in Accuracy')
        ax[3].set_xlabel('Trial Number')
        ax[3].set_ylabel('Accuracy Difference')
        ax[3].legend()
        '''

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_2A_Accuracy_Across_Learning.png')

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
        plt.savefig('SOMA_AL/plots/Figure_2B_Transfer_Choice_Rate.png')

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
    file_names = [r'v1a_avoid_pain\v1a_avoid_pain.csv', r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
    #file_names = [r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv']
    SOMA_pipeline.load_data(file_path=r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific', file_name = file_names)
    
    #Process data
    SOMA_pipeline.process_data()
    #SOMA_pipeline.save_processed_data()

    #Compute summary statistics
    SOMA_pipeline.summarize_data()
    SOMA_pipeline.groupby_summary('group_code')

    #Test code
    SOMA_pipeline.run_tests()

    #Report data
    SOMA_pipeline.build_report()

    #Compute descriptives
    SOMA_pipeline.compute_learning_accuracy()

    #Debug tag
    print()

if __name__ == '__main__':
    main()
    print()


    