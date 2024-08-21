#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

class SOMAPlotting:

    """
    Class to hold plotting functions for the SOMA project
    """

    def print_plots(self):
        self.plot_clinical_scores()
        self.plot_learning_curves(rolling_mean=self.rolling_mean)
        self.plot_learning_curves(rolling_mean=self.rolling_mean, metric='rt')
        self.plot_learning_curves(rolling_mean=self.rolling_mean, context_type='symbol')
        self.plot_learning_curves(rolling_mean=self.rolling_mean, context_type='symbol', metric='rt')
        self.plot_rt_distributions()
        self.plot_transfer_accuracy()
        self.plot_transfer_accuracy(metric='rt')
        self.plot_neutral_transfer_accuracy()
        self.plot_neutral_transfer_accuracy(metric='rt')

    def compute_n_and_t(self, data, splitting_column):

        #Reset index to allow access to all columns
        data = data.reset_index()

        #Compute the sample size and t-score for each group
        if splitting_column == None:
            sample_sizes = data.shape[0]
            t_scores = stats.t.ppf(0.975, sample_sizes-1)
        else:
            sample_sizes = [data[data[splitting_column] == group].shape[0] for group in data[splitting_column].unique()]
            t_scores = [stats.t.ppf(0.975, s-1) for s in sample_sizes]

        return sample_sizes, t_scores

    def plot_learning_curves(self, rolling_mean=None, context_type='context', metric='accuracy'):

        #Add three sublpots, one for each group (group_code), which shows the average accuracy over trials (trial_number) for each of the two contexts (context_val_name)
        num_subplots = 3 if self.split_by_group == 'pain' else 2
        fig, ax = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))
        for i, group in enumerate(self.group_labels):
            group_data = self.learning_data[self.learning_data[self.group_code] == group]

            #Get descriptive statistics for the group
            sample_size = group_data['participant_id'].nunique() #TODO: FIX THIS
            t_score = stats.t.ppf(0.975, sample_size-1)

            #Rename symbol labels
            if context_type == 'context':
                group_data['symbol_name'] = group_data['symbol_name'].replace({'Reward1': 'Reward',
                                                                                        'Reward2': 'Reward', 
                                                                                        'Punish1': 'Punish',
                                                                                        'Punish2': 'Punish'})
                group_data['symbol_name'] = pd.Categorical(group_data['symbol_name'], categories=['Reward', 'Punish'])
                contexts = ['Reward', 'Punish']
            else:
                group_data['symbol_name'] = pd.Categorical(group_data['symbol_name'], categories=['Reward1', 'Reward2', 'Punish1', 'Punish2'])
                contexts = ['Reward1', 'Reward2', 'Punish1', 'Punish2']

            #Determine information of interest
            trial_index_name = 'trial_number' if context_type == 'context' else 'trial_number_symbol'
            color = ['#B2DF8A', '#FB9A99'] if context_type == 'context' else ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C']
            for context_index, context in enumerate(contexts):
                context_data = group_data[group_data['symbol_name'] == context]
                mean_accuracy = context_data.groupby(trial_index_name)[metric].mean()
                CIs = context_data.groupby(trial_index_name)[metric].sem()*t_score
                if rolling_mean is not None:
                    mean_accuracy = mean_accuracy.rolling(rolling_mean, min_periods=1).mean()
                if context_type == 'context':
                    ax[i].fill_between(mean_accuracy.index, mean_accuracy - CIs, mean_accuracy + CIs, alpha=0.2, color=color[context_index], edgecolor='none')
                ax[i].plot(mean_accuracy, color=color[context_index], label=context)

            if metric == 'accuracy':
                ax[i].set_ylim(40, 100)
            ax[i].set_title(f'{group.capitalize()}')
            ax[i].set_xlabel('Trial Number')
            ax[i].set_ylabel(metric.capitalize() if metric != 'rt' else 'Reaction Time (ms)')
            ax[i].legend(loc='lower right', frameon=False)

        #Save the plot
        save_name = f'SOMA_AL/plots/Figure_N_{metric.capitalize()}_Across_Learning.png'
        save_name = save_name.replace('.png', f'_{context_type}.png')
        plt.savefig(save_name)

        #Close figure
        plt.close()

    def plot_rt_distributions(self):

        #create histograms of reaction times for each self.learning_data and self.transfer_data for each group
        fig, ax = plt.subplots(1, len(self.group_labels), figsize=(5*len(self.group_labels), 5))
        for i, group in enumerate(self.group_labels):
            group_data_learning = self.learning_data[self.learning_data[self.group_code] == group]
            group_data_learning = group_data_learning[~group_data_learning['excluded_rt']]
            group_data_transfer = self.transfer_data[self.transfer_data[self.group_code] == group]
            group_data_transfer = group_data_transfer[~group_data_transfer['excluded_rt']]
            
            #normalize data in histogram
            ax[i].hist(group_data_learning['rt'], bins=20, color='C0', alpha=0.5, label='Learning', density=True)
            ax[i].hist(group_data_transfer['rt'], bins=20, color='C1', alpha=0.5, label='Transfer', density=True)
            ax[i].set_title(f'{group.capitalize()}')
            ax[i].set_xlabel('Reaction Time (ms)')
            ax[i].set_ylabel('Density')
            ax[i].legend(loc='upper right', frameon=False)
            ax[i].ticklabel_format(axis='y', style='sci', scilimits=(4,4))

            #Add a point at the mean of the reaction times for each group
            ax[i].scatter(group_data_learning['rt'].mean(), 0, color='C0', s=100, zorder=10)
            ax[i].scatter(group_data_transfer['rt'].mean(), 0, color='C1', s=100, zorder=10)

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_N_RT_distributions.png')

        #Close figure
        plt.close()

    def raincloud_plot(self, data, ax, t_scores, alpha=0.25):
            
            #Set parameters
            if data.index.nunique() == 2:
                colors = ['#B2DF8A', '#FB9A99']
            elif data.index.nunique() == 3:
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
                ax.hlines(mean_data.loc[factor], factor_index+1-0.4, factor_index+1+0.4, color='darkgrey')            

    def plot_transfer_accuracy(self, metric='choice_rate'):

        #Copy choice rate data
        choice_rate = self.choice_rate if metric == 'choice_rate' else self.choice_rt
        
        #Create a bar plot of the choice rate for each symbol
        num_subplots = 3 if self.split_by_group == 'pain' else 2
        fig, ax = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))
        for group_index, group in enumerate(self.group_labels):
            group_choice_rate = choice_rate.loc[group].reset_index()
            group_choice_rate['symbol'] = pd.Categorical(group_choice_rate['symbol'], [4, 3, 2, 1, 0])
            group_choice_rate = group_choice_rate.sort_values('symbol')

            #Compute t-statistic
            _, t_scores = self.compute_n_and_t(group_choice_rate, 'symbol')

            #Get descriptive statistics for the group
            metric_label = 'choice_rate' if metric == 'choice_rate' else 'choice_rt'
            group_choice_rate = group_choice_rate.set_index('symbol')[metric_label].astype(float)

            #Create plot
            self.raincloud_plot(data=group_choice_rate, ax=ax[group_index], t_scores=t_scores)

            #Create horizontal line for the mean the same width
            ax[group_index].set_xticks([1, 2, 3, 4, 5], ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'])
            ax[group_index].set_xlabel('')
            ax[group_index].set_ylabel('Choice Rate (%)' if metric == 'choice_rate' else 'Reaction Time (ms)')
            if metric == 'choice_rate':
                ax[group_index].set_ylim(-4, 104)
            ax[group_index].set_title(group.capitalize())

        #Save the plot
        plt.savefig(f'SOMA_AL/plots/Figure_N_Transfer_{metric}.png')

        #Close figure
        plt.close()

    def plot_neutral_transfer_accuracy(self, metric='choice_rate'):

        #Copy choice rate data
        choice_rate = self.neutral_choice_rate if metric == 'choice_rate' else self.neutral_choice_rt
        choice_rate = choice_rate.reset_index()
        
        #Create a bar plot of the choice rate for each symbol

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        choice_rate['symbol'] = pd.Categorical(choice_rate['group'], self.group_labels)

        #Compute t-statistic
        _, t_scores = self.compute_n_and_t(choice_rate, 'group')

        #Get descriptive statistics for the group
        metric_label = 'choice_rate' if metric == 'choice_rate' else 'choice_rt'
        choice_rate = choice_rate.set_index('group')[metric_label].astype(float)

        #Create plot
        self.raincloud_plot(data=choice_rate, ax=ax, t_scores=t_scores)

        #Create horizontal line for the mean the same width
        x_indexes = [1, 2, 3] if self.split_by_group == 'pain' else [1, 2]
        x_labels = ['No\nPain', 'Acute\nPain', 'Chronic\nPain'] if self.split_by_group == 'pain' else ['Healthy', 'Depressed']
        ax.set_xticks(x_indexes, x_labels)
        ax.set_xlabel('')
        ax.set_ylabel('Choice Rate (%)' if metric == 'choice_rate' else 'Reaction Time (ms)')
        if metric == 'choice_rate':
            ax.set_ylim(-4, 104)
            ax.axhline(y=50, color='darkgrey', linestyle='--')

            #Add vertical annotations on the top and bottom of the plot near the y-axis that says 'Reward' (on the top) and 'Punish' (on the bottom)
            ax.annotate('Reward', xy=(0.55, 95), xytext=(0.55, 95), rotation=90, textcoords='data', ha='center', va='center', color='darkgrey')
            ax.annotate('Punish', xy=(0.55, 5), xytext=(0.55, 5), rotation=90, textcoords='data', ha='center', va='center', color='darkgrey')

        #Save the plot
        plt.savefig(f'SOMA_AL/plots/Figure_N_Neutral_Transfer_{metric}.png')

        #Close figure
        plt.close()

    def plot_clinical_scores(self):

        #Organize clinical data
        metrics = ['intensity', 'unpleasant', 'interference']
        clinical_data = self.pain_scores
        if self.depression_scores is not None:
            clinical_data = clinical_data.merge(self.depression_scores[['participant_id','PHQ8']], on='participant_id', how='outer')
            metrics.append('PHQ8')
        clinical_data = clinical_data.melt(id_vars=[self.group_code, 'participant_id'], value_vars=metrics, var_name='metric', value_name='metric_value')
        clinical_data[self.group_code] = pd.Categorical(clinical_data[self.group_code], self.group_labels)
        clinical_data = clinical_data.sort_values(self.group_code)
        clinical_data = clinical_data.set_index('metric')        

        #Create a bar plot of the choice rate for each symbol
        number_metrics = 4 if self.depression_scores is not None else 3
        fig, ax = plt.subplots(1, number_metrics, figsize=(5*number_metrics, 5))
        for metric_index, metric in enumerate(metrics):
            metric_scores = clinical_data.loc[metric].set_index(self.group_code)['metric_value'].astype(float)

            #Compute t-statistic
            _, t_scores = self.compute_n_and_t(metric_scores, self.group_code)

            #Create plot
            self.raincloud_plot(data=metric_scores, ax=ax[metric_index], t_scores=t_scores, alpha=0.5)

            #Create horizontal line for the mean the same width
            num_groups = 3 if self.split_by_group == 'pain' else 2
            ax[metric_index].set_xticks(list(np.arange(1,num_groups+1)), self.group_labels_formatted)
            ax[metric_index].set_xlabel('')
            ax[metric_index].set_ylabel('Score')
            #ax[metric_index].set_ylim(-4, 104)
            ax[metric_index].set_title(metric.capitalize())

        #Save the plot
        plt.savefig('SOMA_AL/plots/Figure_N_Clinical_Scores.png')

        #Close figure
        plt.close()
        
