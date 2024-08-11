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
        self.plot_clinical_scores()
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
        plt.savefig('SOMA_AL/plots/Figure_N_Accuracy_Across_Learning.png')

        #Close figure
        plt.close()

    def raincloud_plot(self, data, ax, t_scores, alpha=0.25):
            
            #Set parameters
            if data.index.nunique() == 3:
                colors = ['#B2DF8A', '#FFD92F', '#FB9A99']
            else:
                colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']

            #Set index name
            data.index.name = 'code'
            #Turn series into dataframe
            data = data.to_frame()
            data.columns = ['score']

            #Create a violin plot of the data for each level
            wide_data = data.reset_index().pivot(columns='code', values='score')
            wide_list = [wide_data[code].dropna() for code in wide_data.columns]
            vp = ax.violinplot(wide_list, showmeans=False, showmedians=False, showextrema=False)
            
            for bi, b in enumerate(vp['bodies']):
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color(colors[bi])

            #Add jittered scatter plot of the choice rate for each column
            for factor_index, factor in enumerate(data.index.unique()):
                x = np.random.normal([factor_index+1-.2]*data.loc[factor].shape[0], 0.02)
                ax.scatter(x+.02, data.loc[factor], color=colors[factor_index], s=10, alpha=alpha)
            
            #Compute the mean and 95% CIs for the choice rate for each symbol
            mean_data = data.groupby('code').mean()
            CIs = data.groupby('code').sem()['score'] * t_scores

            #Draw rectangle for each symbol that rerpesents the top and bottom of the 95% CI that has no fill and a black outline
            for factor_index, factor in enumerate(data.index.unique()):
                ax.add_patch(plt.Rectangle((factor_index+1-0.4, (mean_data.loc[factor] - CIs.loc[factor])['score']), 0.8, 2*CIs.loc[factor], fill=None, edgecolor='darkgrey'))

            #Add a horizontal **line** for the mean for each factor that is the same width as the 95% CI and is darkgrey
            ax.hlines(mean_data, [h+1-0.4 for h in range(data.index.nunique())], [h+1+0.4 for h in range(data.index.nunique())], color='darkgrey')

    def plot_transfer_accuracy(self):

        #Copy choice rate data
        choice_rate = self.choice_rate
        
        #Create a bar plot of the choice rate for each symbol
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for group_index, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
            group_choice_rate = choice_rate.loc[group].reset_index()
            group_choice_rate['symbol'] = pd.Categorical(group_choice_rate['symbol'], [4, 3, 2, 1, 0])
            group_choice_rate = group_choice_rate.sort_values('symbol')

            sample_size = group_choice_rate['participant'].nunique()
            t_scores = [stats.t.ppf(0.975, sample_size-1)]*5

            #Get descriptive statistics for the group
            group_choice_rate = group_choice_rate.set_index('symbol')['choice_rate'].astype(float)

            #Create plot
            self.raincloud_plot(data=group_choice_rate, ax=ax[group_index], t_scores=t_scores)

            #Create horizontal line for the mean the same width
            ax[group_index].set_xticks([1, 2, 3, 4, 5], ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'])
            ax[group_index].set_xlabel('')
            ax[group_index].set_ylabel('Choice Rate (%)')
            ax[group_index].set_ylim(-4, 104)
            ax[group_index].set_title(group.capitalize())

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_N_Transfer_Choice_Rate.png')

        #Close figure
        plt.close()

    def plot_clinical_scores(self):

        #Organize clinical data
        metrics = ['intensity', 'unpleasant', 'interference']
        clinical_data = self.pain_scores
        if self.depression_scores is not None:
            clinical_data = clinical_data.merge(self.depression_scores[['participant_id','PHQ8']], on='participant_id', how='outer')
            metrics.append('PHQ8')
        clinical_data = clinical_data.melt(id_vars=['group_code', 'participant_id'], value_vars=metrics, var_name='metric', value_name='metric_value')
        clinical_data['group_code'] = pd.Categorical(clinical_data['group_code'], ["no pain", "acute pain", "chronic pain"])
        clinical_data = clinical_data.sort_values('group_code')
        clinical_data = clinical_data.set_index('metric')        

        #Create a bar plot of the choice rate for each symbol
        number_metrics = 4 if self.depression_scores is not None else 3
        fig, ax = plt.subplots(1, number_metrics, figsize=(5*number_metrics, 5))
        for metric_index, metric in enumerate(metrics):
            metric_scores = clinical_data.loc[metric].set_index('group_code')['metric_value'].astype(float)
            #metric_scores = metric_scores.reindex(['no pain', 'acute pain', 'chronic pain'])
            sample_sizes = [len(metric_scores.loc[group]) for group in metric_scores.index.unique()]
            t_scores = [stats.t.ppf(0.975, s-1) for s in sample_sizes]

            #Create plot
            self.raincloud_plot(data=metric_scores, ax=ax[metric_index], t_scores=t_scores, alpha=0.5)

            #Create horizontal line for the mean the same width
            ax[metric_index].set_xticks([1,2,3], ['No\nPain', 'Acute\nPain', 'Chronic\nPain'])
            ax[metric_index].set_xlabel('')
            ax[metric_index].set_ylabel('Score')
            #ax[metric_index].set_ylim(-4, 104)
            ax[metric_index].set_title(metric.capitalize())

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_N_Clinical_Scores.png')

        #Close figure
        plt.close()
        
