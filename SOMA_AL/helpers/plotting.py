#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image

class Plotting:

    """
    Class to hold plotting functions for the SOMA project
    """

    def __init__(self):
        self.colors = {'group': ['#B2DF8A', '#FFD92F', '#FB9A99'],
                       'condition': ['#095086', '#9BD2F2', '#ECA6A6', '#B00000', '#D3D3D3'],
                       'condition_2': ['#9BD2F2', '#ECA6A6']}
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 14
                                      
    #Helper functions
    def print_plots(self) -> None:

        """
        Print all plots as images
        """

        #Main figures
        self.plot_clinical_scores('demo-clinical-scores', colors=self.colors['group'])
        self.plot_learning_curves('learning-accuracy-by-group', rolling_mean=self.rolling_mean, grouping='clinical', colors=self.colors['condition_2'])
        self.plot_learning_curves('learning-rt-by-group', rolling_mean=self.rolling_mean, grouping='clinical', metric='rt', colors=self.colors['condition_2'])
        self.plot_transfer_data('transfer-choice-rate', colors=self.colors['condition'], plot_type='bar', group_labels=False)
        self.plot_transfer_data('transfer-rt', colors=self.colors['condition'], plot_type='bar', group_labels=False)
        self.plot_combined_learning_and_transfer('empirical-performance', 'learning-accuracy-by-group', 'transfer-choice-rate')
        self.plot_combined_learning_and_transfer('empirical-rt', 'learning-rt-by-group', 'transfer-rt')

        #Other figures
        self.plot_learning_curves('learning-accuracy-by-context', rolling_mean=self.rolling_mean, grouping='context', colors=self.colors['group'])
        self.plot_learning_curves('learning-rt-by-context', rolling_mean=self.rolling_mean, grouping='context', metric='rt', colors=self.colors['group'])
        self.plot_transfer_data('learning-accuracy', colors=self.colors['condition_2'])
        self.plot_transfer_data('learning-accuracy-context', colors=self.colors['group'])
        self.plot_transfer_data('learning-accuracy-diff', colors=self.colors['group'])
        self.plot_transfer_data('learning-accuracy-context-diff', colors=self.colors['group'])
        self.plot_transfer_data('learning-rt', colors=self.colors['condition_2'])
        self.plot_transfer_data('learning-rt-context', colors=self.colors['group'])
        self.plot_transfer_data('learning-rt-diff', colors=self.colors['group'])
        self.plot_transfer_data('learning-rt-context-diff', colors=self.colors['group'])
        self.plot_transfer_data('transfer-choice-rate', colors=self.colors['condition'])
        self.plot_transfer_data('transfer-rt', colors=self.colors['condition'])
        self.plot_transfer_data('transfer-valence-bias', colors=self.colors['group'])
        self.plot_select_transfer('select-choice-rate', colors=self.colors['condition'], plot_type='bar')
        self.plot_select_transfer('select-choice-rate', colors=self.colors['condition'])
        self.plot_neutral_transfer_accuracy('transfer-choice-rate-neutral', colors=self.colors['group'])
        self.plot_neutral_transfer_accuracy('transfer-rt-neutral', metric='rt', colors=self.colors['group'])
        self.plot_differences_transfer('transfer-choice-rate-differences', colors=self.colors['group'])

    def compute_n_and_t(self, data: pd.DataFrame, splitting_column: str) -> tuple:

        """
        Compute the sample size and t-score for each group

        Parameters
        ----------
        data : DataFrame
            The data to be analyzed
        splitting_column : str
            The column to split the data by

        Returns
        -------
        sample_sizes : list
            The sample sizes for each group
        t_scores : list
            The t-scores for each group
        """

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

    #Plotting functions
    def raincloud_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.5, colors: list = []) -> None:
            
            """
            Create a raincloud plot of the data

            Parameters
            ----------
            data : DataFrame
                The data to be plotted
            ax : Axes
                The axes to plot the data on
            t_scores : list
                The t-scores for each group
            alpha : float
                The transparency of the scatter plot
            """

            #Set index name
            data.index.name = 'code'
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
            CIs = data.groupby('code').sem()['score'] 
            CIs = CIs.dropna()
            CIs = CIs * t_scores

            #Draw rectangle for each symbol that rerpesents the top and bottom of the 95% CI that has no fill and a black outline
            for factor_index, factor in enumerate(data.index.unique()):
                ax.add_patch(plt.Rectangle((factor_index+1-0.4, (mean_data.loc[factor] - CIs.loc[factor])['score']), 0.8, 2*CIs.loc[factor], fill=None, edgecolor='darkgrey'))
                ax.hlines(mean_data.loc[factor], factor_index+1-0.4, factor_index+1+0.4, color='darkgrey')      

    def bar_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.5, colors: list = []) -> None:
            
            """
            Create a raincloud plot of the data

            Parameters
            ----------
            data : DataFrame
                The data to be plotted
            ax : Axes
                The axes to plot the data on
            t_scores : list
                The t-scores for each group
            alpha : float
                The transparency of the scatter plot
            """

            #Set index name
            data.index.name = 'code'
            data = data.to_frame()
            data.columns = ['score']
            
            #Compute the mean and 95% CIs for the choice rate for each symbol
            mean_data = data.groupby('code').mean()
            mean_data = mean_data.dropna()
            CIs = data.groupby('code').sem()['score'] 
            CIs = CIs.dropna()
            CIs = CIs * t_scores

            #Add barplot with CIs
            ax.bar(np.arange(1,len(mean_data['score'])+1), mean_data['score'], yerr=CIs, color=colors, alpha=alpha, capsize=5, ecolor='dimgrey')                        

    def plot_combined_learning_and_transfer(self, save_name: str, learning_name: str, transfer_name: str, image_height: int = 1000) -> None:
        """
        Combine separately saved learning and transfer plots into a single stacked image.

        Parameters
        ----------
        save_name : str
            The filename (no extension) to save the combined image as.
        learning_args : dict
            Arguments to pass to plot_learning_curves (must include 'save_name').
        transfer_args : dict
            Arguments to pass to plot_transfer_data (must include 'save_name').
        image_height : int
            Desired height (in pixels) of each subplot image.
        """

        path = f'SOMA_AL/plots/{self.split_by_group}'
        learning_path = f'{path}/{learning_name}.png'
        transfer_path = f'{path}/{transfer_name}.png'

        # Open the images and resize to have the same width
        learning_img = Image.open(learning_path)
        transfer_img = Image.open(transfer_path)

        # Resize images to have the same width (whichever is smaller)
        width = min(learning_img.width, transfer_img.width)
        combined_img = Image.new('RGB', (width, learning_img.height*2))
        combined_img.paste(learning_img, (0, 0))
        combined_img.paste(transfer_img, (0, learning_img.height))

        # Save combined image
        combined_path = f'SOMA_AL/plots/{self.split_by_group}/{save_name}.png'
        combined_img.save(combined_path)

    def plot_learning_curves(self, save_name: str, rolling_mean: int = None,  metric: str = 'accuracy', grouping: str = 'clinical', colors: list = []) -> None:

        """
        Plot the learning curves for the accuracy or reaction time data

        Parameters
        ----------
        save_name : str
            The name to save the plot as
        rolling_mean : int
            The number of trials to average over
        metric : str
            The metric to plot (y)
        grouping : str
            The grouping to plot the data by

        Returns (External)
        ------------------
        Image: PNG
            A plot of the learning curves
        """

        #Set grouping parameters
        if grouping == 'clinical':
            grouping_labels = self.group_labels
            grouping_code = self.group_code
            contexts = ['Reward', 'Punish']
            contexts_code = 'symbol_name'
        else:
            grouping_labels = ['Reward', 'Punish']
            grouping_code = 'symbol_name'
            contexts = ['no pain', 'chronic pain'] if self.split_by_group == 'pain' else ['healthy', 'depressed']
            contexts_code = self.group_code

        #Add three sublpots, one for each group (group_code), which shows the average accuracy over trials (trial_number) for each of the two contexts (context_val_name)
        num_subplots = len(grouping_labels)
        fig, ax = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))
        for i, group in enumerate(grouping_labels):
            group_data = self.learning_data[self.learning_data[grouping_code] == group]

            #Get descriptive statistics for the group
            sample_size = group_data['participant_id'].nunique() #TODO: FIX THIS, USE FUNCTION
            t_score = stats.t.ppf(0.975, sample_size-1)

            #Average duplicate trial_numbers for each participant within each symbol_name but then also keep the symbol_name column
            group_data = group_data[['participant_id', 'trial_number', contexts_code, metric]]
            group_data = group_data.groupby(['participant_id', 'trial_number', contexts_code]).mean().reset_index()

            #Determine information of interest
            trial_index_name = 'trial_number'
            for context_index, context in enumerate(contexts):
                context_data = group_data[group_data[contexts_code] == context]
                mean_accuracy = context_data.groupby(trial_index_name)[metric].mean()
                CIs = context_data.groupby(trial_index_name)[metric].sem()*t_score
                if rolling_mean is not None:
                    mean_accuracy = mean_accuracy.rolling(rolling_mean, min_periods=1, center=True).mean()
                ax[i].fill_between(mean_accuracy.index, mean_accuracy - CIs, mean_accuracy + CIs, alpha=0.2, color=colors[context_index], edgecolor='none')
                ax[i].plot(mean_accuracy, color=colors[context_index], label=context.title(), linewidth=3)

            if metric == 'accuracy':
                ax[i].set_ylim(40, 100)
            ax[i].set_title(f'{group.capitalize()}')
            ax[i].set_xlabel('Trial Number')
            ax[i].set_ylabel(metric.capitalize() if metric != 'rt' else 'Reaction Time (ms)')
            ax[i].legend(loc='lower right', frameon=False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].tick_params(axis='both')   


        #Save the plot
        plt.tight_layout()
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def plot_transfer_data(self, save_name: str, colors: list, plot_type: str = 'raincloud', group_labels: bool = True) -> None:

        """
        Create raincloud plots of the data

        Parameters
        ----------
        save_name : str
            The name to save the plot as

        Returns (External)
        ------------------
        Image: PNG
            A plot of the raincloud plots
        """

        #Set data specific parameters
        match save_name:
            case 'learning-accuracy' | 'learning-accuracy-context':
                data = self.learning_accuracy
                metric_label = 'accuracy'
                y_label = 'Accuracy (%)'
            case 'learning-accuracy-diff' | 'learning-accuracy-context-diff':
                data = self.learning_accuracy_diff
                metric_label = 'accuracy'
                y_label = 'Difference in Accuracy (%), Reward - Punish'
            case 'learning-rt' | 'learning-rt-context':
                data = self.learning_rt
                metric_label = 'rt'
                y_label = 'Reaction Time (ms)'
            case 'learning-rt-diff' | 'learning-rt-context-diff':
                data = self.learning_rt_diff
                metric_label = 'rt'
                y_label = 'Difference in Reaction Time (ms), Reward - Punish'    
            case 'transfer-choice-rate':
                data = self.choice_rate
                data = data.reset_index()
                data['symbol'] = data['symbol'].replace({'Novel': 0, 'High Reward': 4, 'Low Reward': 3, 'Low Punish': 2, 'High Punish': 1})
                data = data.set_index([self.group_code, 'participant_id', 'symbol'])
                metric_label = 'choice_rate'
                y_label = 'Choice Rate (%)'
            case 'transfer-rt':
                data = self.choice_rt
                data = data.reset_index()
                data['symbol'] = data['symbol'].replace({'Novel': 0, 'High Reward': 4, 'Low Reward': 3, 'Low Punish': 2, 'High Punish': 1})
                data = data.set_index([self.group_code, 'participant_id', 'symbol'])
                metric_label = 'choice_rt'
                y_label = 'Reaction Time (ms)'
            case 'transfer-valence-bias':
                data = self.valence_bias
                metric_label = 'valence_bias'
                y_label = 'Accuracy Bias (%)'

        if 'context' in save_name:
            data = data.reset_index().set_index(['symbol_name', self.group_code, 'participant_id'])

        if 'select-choice-rate-' in save_name:
            symbol = save_name.split('-')[-1]
            data = self.select_choice_rate[symbol]
            data = data.reset_index()
            data['symbol'] = data['symbol'].replace({'Novel': 0, 'High Reward': 4, 'Low Reward': 3, 'Low Punish': 2, 'High Punish': 1})
            data = data.set_index([self.group_code, 'participant_id', 'symbol'])
            metric_label = 'choice_rate'
            y_label = 'Choice Rate (%)'

        if 'diff' in save_name or 'context' in save_name or 'valence-bias' in save_name:
            condition_name = self.group_code
            condition_values = self.group_labels
            x_values = np.arange(1, len(self.group_labels)+1).tolist()
            x_labels = self.group_labels_formatted
        elif save_name == 'learning-accuracy' or save_name == 'learning-rt':
            condition_name = 'symbol_name'
            condition_values = ['Reward', 'Punish']
            x_values = [1, 2]
            x_labels = ['Reward', 'Punish']
        else:
            condition_name = 'symbol'
            condition_values = [4, 3, 2, 1, 0]
            x_values = [1, 2, 3, 4, 5]
            x_labels = ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel']

        if 'diff' in save_name or 'valence-bias' in save_name:
            plot_labels = ['']
        elif 'context' in save_name:
            plot_labels = ['Reward', 'Punish']
        else:
            plot_labels = self.group_labels
        
        #Create a bar plot of the choice rate for each symbol
        if 'diff' in save_name or 'valence-bias' in save_name:
            num_subplots = 1
        elif 'context' in save_name:
            num_subplots = 2
        else:
            num_subplots = 3 if self.split_by_group == 'pain' else 2
        
        fig, ax = plt.subplots(1, num_subplots, figsize=(5*num_subplots, 5))
        for group_index, group in enumerate(plot_labels):
            if group != '':
                group_data = data.loc[group].reset_index()
            else:
                group_data = data.reset_index()
            group_data[condition_name] = pd.Categorical(group_data[condition_name], condition_values)
            group_data = group_data.sort_values(condition_name)

            #Compute t-statistic
            _, t_scores = self.compute_n_and_t(group_data, condition_name)

            #Get descriptive statistics for the group
            group_data = group_data.set_index(condition_name)[metric_label].astype(float)

            #Create plot
            if plot_type == 'raincloud':
                if 'diff' not in save_name and 'valence-bias' not in save_name:
                    self.raincloud_plot(data=group_data, ax=ax[group_index], t_scores=t_scores, colors=colors)
                else:
                    self.raincloud_plot(data=group_data, ax=ax, t_scores=t_scores, colors=colors)
            elif plot_type == 'bar':
                if 'diff' not in save_name and 'valence-bias' not in save_name:
                    self.bar_plot(data=group_data, ax=ax[group_index], t_scores=t_scores, colors=colors)
                else:
                    self.bar_plot(data=group_data, ax=ax, t_scores=t_scores, colors=colors)      
            else:
                raise ValueError(f'Plot type {plot_type} not recognized.')          

            #Create horizontal line for the mean the same width
            if 'diff' not in save_name and 'valence-bias' not in save_name:
                ax[group_index].set_xticks(x_values, x_labels)
                ax[group_index].set_xlabel('')
                ax[group_index].set_ylabel(y_label)
                if '(%)' in y_label:
                    ylim = [-4, 104] if plot_type == 'raincloud' else [0, 100]
                    ax[group_index].set_ylim(ylim)
                if group_labels:
                    ax[group_index].set_title(group.capitalize())
                else:
                    ax[group_index].set_title('')
                ax[group_index].spines['top'].set_visible(False)
                ax[group_index].spines['right'].set_visible(False)
                ax[group_index].tick_params(axis='both')   

            else:
                ax.set_xticks(x_values, x_labels)
                ax.set_xlabel('')
                ax.set_ylabel(y_label)
                ax.set_title('')
                ax.axhline(y=0, color='darkgrey', linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both')   

        #Save the plot
        plt.tight_layout()
        save_name = f'{save_name}_supplemental' if plot_type == 'raincloud' else save_name
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def plot_differences_transfer(self, save_name: str, colors: list) -> None:

        data = self.choice_rate
        data = data.reset_index()
        data = data[data['symbol'].isin(['Low Reward', 'Low Punish'])]
        data = data.pivot(index=['participant_id', self.group_code], columns='symbol', values='choice_rate')
        data.reset_index(inplace=True)
        data['difference'] = data.apply(lambda x: x['Low Reward'] - x['Low Punish'], axis=1)

        #Create a raincloud plot the difference for each group code
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _, t_scores = self.compute_n_and_t(data, self.group_code)
        #Categorize the group code no pain, acute pain, chronic pain
        data[self.group_code] = pd.Categorical(data[self.group_code], categories=self.group_labels, ordered=True)
        data = data.set_index(self.group_code)['difference'].astype(float)
        data.index = pd.CategoricalIndex(data.index, categories=self.group_labels, ordered=True)
        data = data.sort_index()
        self.raincloud_plot(data=data, ax=ax, t_scores=t_scores, colors=colors)

        ax.set_xticks([1, 2, 3], ['No Pain', 'Acute Pain', 'Chronic Pain'])
        ax.set_xlabel('')
        ax.set_ylabel('Difference in Choice Rate, Low Reward - Low Punish (%)')
        ax.axhline(y=0, color='darkgrey', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-100, 100)
        ax.tick_params(axis='both')

        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.png')
        plt.close()

    def plot_select_transfer(self, save_name: str, colors: list, plot_type: str = 'raincloud') -> None:
        """
        Create raincloud plots of the data

        Parameters
        ----------
        save_name : str
            The name to save the plot as

        Returns (External)
        ------------------
        Image: PNG
            A plot of the raincloud plots
        """

        #Set data specific parameters
        metric_label = 'choice_rate'
        y_label = 'Choice Rate (%)'
        condition_name = 'symbol'
        condition_values = [4, 3, 2, 1, 0]
        x_ids = ['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel']
        x_labels = ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel']
        plot_labels = self.group_labels
        
        #Create a bar plot of the choice rate for each symbol
        fig, ax = plt.subplots(5, 3, figsize=(15, 20))
        fig.subplots_adjust(hspace=0.35)
        for symbol_index, symbol in enumerate(['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel']):
            data = self.select_choice_rate[symbol]
            data = data.reset_index()
            data['symbol'] = data['symbol'].replace({'Novel': 0, 'High Reward': 4, 'Low Reward': 3, 'Low Punish': 2, 'High Punish': 1})
            data = data.set_index([self.group_code, 'participant_id', 'symbol'])
            symbol_x_ids = [x_id for x_id in x_ids if x_id != symbol]
            symbol_x_labels = [x_label for x_label in x_labels if x_label.replace('\n',' ') != symbol]
            symbol_colours = [colors[x_ids.index(x_id)] for x_id in symbol_x_ids]
            
            for group_index, group in enumerate(plot_labels):
                group_data = data.loc[group].reset_index()
                group_data[condition_name] = pd.Categorical(group_data[condition_name], condition_values)
                group_data = group_data.sort_values(condition_name)

                #Compute t-statistic
                _, t_scores = self.compute_n_and_t(group_data, condition_name)

                #Get descriptive statistics for the group
                group_data = group_data.set_index(condition_name)[metric_label].astype(float)

                #Create plot
                if plot_type == 'raincloud':
                    self.raincloud_plot(data=group_data, ax=ax[symbol_index, group_index], t_scores=t_scores, colors=symbol_colours)        
                elif plot_type == 'bar':
                    self.bar_plot(data=group_data, ax=ax[symbol_index, group_index], t_scores=t_scores, colors=symbol_colours)
                else:
                    raise ValueError(f'Plot type {plot_type} not recognized.')            

                #Create horizontal line for the mean the same width
                ax[symbol_index, group_index].set_xticks([1,2,3,4], symbol_x_labels)
                ax[symbol_index, group_index].set_xlabel('')
                ax[symbol_index, group_index].axhline(y=50, color='darkgrey', linestyle='--', alpha=0.5)
                if group_index == 0:
                    ax[symbol_index, group_index].set_ylabel(f'{symbol.title()}\n{y_label}')
                if '(%)' in y_label:
                    ax[symbol_index, group_index].set_ylim(-4, 104)
                if symbol_index == 0:
                    ax[symbol_index, group_index].set_title(group.capitalize())
                ax[symbol_index, group_index].tick_params(axis='both')   

        #Save the plot
        save_name = f'{save_name}_supplemental' if plot_type == 'raincloud' else save_name
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/selected_{save_name}.png')
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/selected_{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def plot_neutral_transfer_accuracy(self, save_name: str, metric: str = 'choice_rate', colors: list = []) -> None:

        """
        Plot the neutral transfer accuracy data
        
        Parameters
        ----------
        save_name : str
            The name to save the plot as
        metric : str
            The metric (y) to be plotted plot (choice_rate or choice_rt)

        Returns (External)
        ------------------
        Image: PNG
            A plot of the raincloud plots
        """

        #Copy choice rate data
        choice_rate = self.neutral_choice_rate if metric == 'choice_rate' else self.neutral_choice_rt
        choice_rate = choice_rate.reset_index()
        
        #Create a bar plot of the choice rate for each symbol

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        choice_rate['symbol'] = pd.Categorical(choice_rate[self.group_code], self.group_labels)

        #Compute t-statistic
        _, t_scores = self.compute_n_and_t(choice_rate, self.group_code)

        #Get descriptive statistics for the group
        metric_label = 'choice_rate' if metric == 'choice_rate' else 'choice_rt'
        choice_rate = choice_rate.set_index(self.group_code)[metric_label].astype(float)

        #Create plot
        self.raincloud_plot(data=choice_rate, ax=ax, t_scores=t_scores, colors=colors)

        #Create horizontal line for the mean the same width
        x_indexes = [1, 2, 3] if self.split_by_group == 'pain' else [1, 2]
        x_labels = ['No\nPain', 'Acute\nPain', 'Chronic\nPain'] if self.split_by_group == 'pain' else ['Healthy', 'Depressed']
        ax.set_xticks(x_indexes, x_labels)
        ax.set_xlabel('')
        ax.set_ylabel('Choice Rate (%)' if metric == 'choice_rate' else 'Reaction Time (ms)')
        ax.tick_params(axis='both')   

        if metric == 'choice_rate':
            ax.set_ylim(-4, 104)
            ax.axhline(y=50, color='darkgrey', linestyle='--')

            #Add vertical annotations on the top and bottom of the plot near the y-axis that says 'Reward' (on the top) and 'Punish' (on the bottom)
            ax.annotate('Reward', xy=(0.55, 95), xytext=(0.55, 95), rotation=90, textcoords='data', ha='center', va='center', color='darkgrey')
            ax.annotate('Punish', xy=(0.55, 5), xytext=(0.55, 5), rotation=90, textcoords='data', ha='center', va='center', color='darkgrey')

        #Save the plot
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def plot_clinical_scores(self, save_name: str, colors: list) -> None:

        """
        Plot the clinical scores for the participants

        Parameters
        ----------
        save_name : str
            The name to save the plot as

        Returns (External)
        ------------------
        Image: PNG
            A plot of the raincloud plots
        """

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
            self.raincloud_plot(data=metric_scores, ax=ax[metric_index], t_scores=t_scores, alpha=0.5, colors=colors)

            #Create horizontal line for the mean the same width
            num_groups = 3 if self.split_by_group == 'pain' else 2
            ax[metric_index].set_xticks(list(np.arange(1,num_groups+1)), self.group_labels_formatted)
            ax[metric_index].set_xlabel('')
            ax[metric_index].set_ylabel('Score')
            ax[metric_index].set_title(metric.capitalize())
            ax[metric_index].spines['top'].set_visible(False)
            ax[metric_index].spines['right'].set_visible(False)
            ax[metric_index].set_ylim(0, 10)
            ax[metric_index].tick_params(axis='both') 

        #Save the plot
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'SOMA_AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()
            
    def plot_model_parameters_by_pain(self, fit_data: pd.DataFrame, parameter_names: list, pain_names: list) -> None:
        fig, axes = plt.subplots(nrows=len(parameter_names), ncols=len(pain_names), figsize=(len(pain_names)*3, len(parameter_names)*3))
        colours = ['#B2DF8A', '#FB9A99', '#FFD92F']
        for i, parameter in enumerate(parameter_names):
            for j, pain_metric in enumerate(pain_names):
                rs = []
                ps = []
                ns = []
                for gi, group in enumerate(['no pain', 'chronic pain', 'acute pain']):
                    group_data = fit_data[fit_data['pain_group'] == group]
                    correlation_data = group_data[[parameter, pain_metric]]
                    x_min, x_max = correlation_data[pain_metric].min(), correlation_data[pain_metric].max()
                    y_min, y_max = correlation_data[parameter].min(), correlation_data[parameter].max()
                    r, p = stats.pearsonr(correlation_data[parameter], correlation_data[pain_metric])
                    rs.append(r)
                    ps.append(p)
                    ns.append(len(correlation_data))
                    axes[i,j].scatter(correlation_data[pain_metric], correlation_data[parameter], alpha=0.3, color=colours[gi], s=10, label=group)
                    slope, intercept, _, _, _ = stats.linregress(correlation_data[pain_metric], correlation_data[parameter])
                    axes[i,j].plot(correlation_data[pain_metric], slope*correlation_data[pain_metric] + intercept, color=colours[gi], linewidth=1.5, label=None)

                #Set titles only on top row
                if i == 0:
                    axes[i,j].set_title(pain_metric.title(), fontsize=10)
                if j == 0:
                    y_label = parameter.replace('_', ' ').replace('lr', 'learning rate').title()
                    axes[i,j].set_ylabel(y_label, fontsize=10)
                
                axes[i,j].set_xlabel('')
                x_min, x_max = fit_data[pain_metric].min(), fit_data[pain_metric].max()
                y_min, y_max = np.log(fit_data[parameter]+1.01).min(), np.log(fit_data[parameter]+1).max()
                axes[i,j].set_xlim(x_min, x_max)
                axes[i,j].set_ylim(y_min, y_max)
                axes[i,j].tick_params(axis='both', which='major', labelsize=8)
                axes[i,j].tick_params(axis='both', which='minor', labelsize=6)
                #Add legend above the last column in the first row
                if j == len(pain_names) - 1 and i == 0:
                    axes[i,j].legend(['No Pain', 'Chronic Pain', 'Acute Pain'], loc='upper left', fontsize=8, frameon=False, bbox_to_anchor=(1.05, 1))

        plt.savefig(f'SOMA_AL/plots/model_parameter_by_pain.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'SOMA_AL/plots/model_parameter_by_pain.svg', format='svg', bbox_inches='tight')

        #Close figure
        plt.close(fig)

    def plot_model_parameters_by_pain_split(self, fit_data: pd.DataFrame, parameter_names: list, pain_names: list):
        colours = {'no pain': '#B2DF8A', 'chronic pain': '#FB9A99', 'acute pain': '#FFD92F'}
        
        for group in ['no pain', 'chronic pain', 'acute pain']:
            fig, axes = plt.subplots(nrows=len(parameter_names), ncols=len(pain_names), 
                                    figsize=(len(pain_names)*3, len(parameter_names)*3))
            
            for i, parameter in enumerate(parameter_names):
                for j, pain_metric in enumerate(pain_names):
                    group_data = fit_data[fit_data['pain_group'] == group]
                    correlation_data = group_data[[parameter, pain_metric]].dropna()
                    
                    x_min, x_max = correlation_data[pain_metric].min(), correlation_data[pain_metric].max()
                    y_min, y_max = correlation_data[parameter].min(), correlation_data[parameter].max()
                    
                    r, p = stats.pearsonr(correlation_data[parameter], correlation_data[pain_metric])
                    
                    axes[i, j].scatter(correlation_data[pain_metric], correlation_data[parameter], 
                                    alpha=0.3, color=colours[group], s=10)
                    
                    slope, intercept, _, _, _ = stats.linregress(correlation_data[pain_metric], 
                                                                correlation_data[parameter])
                    axes[i, j].plot(correlation_data[pain_metric], 
                                    slope * correlation_data[pain_metric] + intercept, 
                                    color=colours[group], linewidth=1.5)
                    
                    if i == 0:
                        axes[i, j].set_title(pain_metric.title(), fontsize=10)
                    if j == 0:
                        y_label = parameter.replace('_', ' ').replace('lr', 'learning rate').title().replace('Learning', '\nLearning')
                        axes[i, j].set_ylabel(y_label, fontsize=10)
                    
                    axes[i, j].set_xlim(x_min, x_max)
                    axes[i, j].set_ylim(y_min, y_max)
                    axes[i, j].tick_params(axis='both', which='major', labelsize=8)
                    axes[i, j].tick_params(axis='both', which='minor', labelsize=6)
            
            plt.savefig(f'SOMA_AL/plots/model_parameter_by_pain_{group.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'SOMA_AL/plots/model_parameter_by_pain_{group.replace(" ", "_")}.svg', format='svg', bbox_inches='tight')

            #Close figure
            plt.close(fig)

    def plot_model_fits(self, fits: dict) -> None:
        group_labels = ['no pain', 'acute pain', 'chronic pain', 'full']

        for fit in fits:
            current_fit = fits[fit].copy().iloc[:,:-1]
            current_fit = current_fit.reset_index()
            current_models = current_fit.columns[1:]
            current_models = [model.replace('ContextualQ', 'w-Relative') for model in current_models]
            number_subplots = len(group_labels)
            max_fit = int(current_fit.iloc[:,1:].to_numpy().max()*1.1)
            
            fig, ax = plt.subplots(1, number_subplots, figsize=(5*number_subplots, 5))
            for i, group in enumerate(group_labels):
                current_values = current_fit.iloc[i,:].values[1:].astype(int)
                current_colors = ['black' if i == np.min(current_values) else 'white' for i in current_values]
                ax[i].bar(current_models,
                          current_values,
                          color = self.colors['group'][i] if i < number_subplots - 1 else 'dimgrey',
                          alpha=0.5,
                          capsize=5,
                          edgecolor=current_colors,
                          linewidth=2)            
                #Add the number above as annotations
                for j, value in enumerate(current_values):
                    ax[i].text(j, value + 0.5, str(value), ha='center', va='bottom', fontsize=8, color='darkgrey')
                ax[i].set_title(group.replace('full','all').title())
                ax[i].set_xlabel('')
                ax[i].set_ylabel(fit)
                ax[i].set_ylim(0, max_fit)
                ax[i].set_xticks(current_models, current_models)
                ax[i].set_xticklabels(current_models, rotation=45, ha='right')
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
               
            #Save the plot
            plt.tight_layout()
            plt.savefig(f'SOMA_AL/modelling/group_{fit}.png')
            plt.close()
