#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class SOMAPlotting:

    """
    Class to hold plotting functions for the SOMA project
    """

    def print_plots(self):
        self.plot_learning_accuracy(rolling_mean=5)
        self.plot_transfer_accuracy()

    def plot_learning_accuracy(self, rolling_mean=None):

        #Track parameters
        self.fig1_rolling_mean = rolling_mean

        #Add three sublpots, one for each group (group_code), which shows the average accuracy over trials (trial_number) for each of the two contexts (context_val_name)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_data = self.learning_data[self.learning_data['group_code'] == group]

            #Get descriptive statistics for the group
            sample_size = group_data['participant_id'].nunique()
            t_score = stats.t.ppf(0.975, sample_size-1)

            for context_index, context in enumerate(['Reward', 'Loss Avoid']):
                context_data = group_data[group_data['context_val_name'] == context]
                mean_accuracy = context_data.groupby('trial_number')['accuracy'].mean()
                CIs = context_data.groupby('trial_number')['accuracy'].sem()*t_score
                if rolling_mean is not None:
                    mean_accuracy = mean_accuracy.rolling(rolling_mean).mean()
                ax[i].fill_between(mean_accuracy.index, mean_accuracy - CIs, mean_accuracy + CIs, alpha=0.2, color=['#B2DF8A', '#FB9A99'][context_index], edgecolor='none')
                ax[i].plot(mean_accuracy, color=['#B2DF8A', '#FB9A99'][context_index], label=['Reward' if context == 'Reward' else 'Punish'])

            ax[i].set_ylim(40, 100)
            ax[i].set_title(f'{group.capitalize()}')
            ax[i].set_xlabel('Trial Number')
            ax[i].set_ylabel('Accuracy')
            ax[i].legend(loc='lower right', frameon=False)

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_1_Accuracy_Across_Learning.png')

        #Close figure
        plt.close()

    def plot_transfer_accuracy(self):

        #Copy choice rate data
        choice_rate = self.choice_rate
        
        #Create a bar plot of the choice rate for each symbol
        colors = ['#D3D3D3', '#E31A1C', '#FB9A99', '#B2DF8A', '#33A02C']
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_choice_rate = choice_rate.loc[group].reset_index()

            #Get descriptive statistics for the group
            sample_size = group_choice_rate['participant'].nunique()
            t_score = stats.t.ppf(0.975, sample_size-1)
            group_choice_rate = group_choice_rate.pivot(index='participant', columns='symbol', values='choice_rate').astype(float)

            vp = ax[i].violinplot(group_choice_rate, showmeans=False, showmedians=False, showextrema=False)
            
            for bi, b in enumerate(vp['bodies']):
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(colors[bi])

            #Add jittered scatter plot of the choice rate for each column
            for symbol in [0, 1, 2, 3, 4]:
                x = np.random.normal([symbol+1+.2]*group_choice_rate.shape[0], 0.02)
                ax[i].scatter(x-.02, group_choice_rate[symbol], color=colors[symbol], s=10, alpha=0.25)
            
            #Compute the mean and 95% CIs for the choice rate for each symbol
            mean_choice_rate = group_choice_rate.mean()
            CIs = group_choice_rate.sem() * t_score

            #Draw rectangle for each symbol that rerpesents the top and bottom of the 95% CI that has no fill and a black outline
            for symbol in [0, 1, 2, 3, 4]:
                ax[i].add_patch(plt.Rectangle((symbol+1-0.4, mean_choice_rate.loc[symbol] - CIs.loc[symbol]), 0.8, 2*CIs.loc[symbol], fill=None, edgecolor='darkgrey'))

            #Add a horizontal **line** for the mean choice rate for each symbol that is the same width as the 95% CI and is darkgrey
            ax[i].hlines(mean_choice_rate, [h-0.4 for h in [1, 2, 3, 4, 5]], [h+0.4 for h in [1, 2, 3, 4, 5]], color='darkgrey')

            #Create horizontal line for the mean the same width
            ax[i].set_xticks([1, 2, 3, 4, 5], ['Novel', 'High\nPunish', 'Low\nPunish', 'Low\nReward', 'High\nReward'])
            ax[i].invert_xaxis()
            ax[i].set_xlabel('')
            ax[i].set_ylabel('Choice Rate (%)')
            ax[i].set_ylim(-4, 104)
            ax[i].set_title(group.capitalize())

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_2_Transfer_Choice_Rate.png')

        #Close figure
        plt.close()
