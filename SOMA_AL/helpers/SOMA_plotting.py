#Import modules
import matplotlib.pyplot as plt

class SOMAPlotting:
    """
    Class to hold plotting functions for the SOMA project
    """

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
