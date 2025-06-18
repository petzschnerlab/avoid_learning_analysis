#Import modules
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

class Plotting:

    """
    Class to hold plotting functions for the SOMA project
    """

    def __init__(self):

        #Initialize plot formatting
        self.colors = {'group': ['#85A947', '#3E7B27', '#123524'],
                       'condition': ['#095086', '#9BD2F2', '#ECA6A6', '#B00000', '#D3D3D3'],
                       'condition_2': ['#095086', '#B00000']}
        
        if 'Helvetica' in set(f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = 'Helvetica'

        plt.rcParams['font.size'] = 18
                                      
    #Helper functions
    def print_plots(self) -> None:

        """
        Print all plots as images
        """

        #Main figures
        self.plot_clinical_scores('demo-clinical-scores', colors=self.colors['group'])
        self.plot_learning_curves('learning-accuracy-by-group', rolling_mean=self.rolling_mean, grouping='clinical', colors=self.colors['condition_2'], subplot_title='A. Learning Phase')
        self.plot_learning_curves('learning-rt-by-group', rolling_mean=self.rolling_mean, grouping='clinical', metric='rt', colors=self.colors['condition_2'], subplot_title='A. Learning Phase')
        self.plot_transfer_data('transfer-choice-rate', colors=self.colors['condition'], plot_type='bar', group_labels=False, subplot_title='B. Transfer Phase')
        self.plot_transfer_data('transfer-rt', colors=self.colors['condition'], plot_type='bar', group_labels=False, subplot_title='B. Transfer Phase')
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
        self.plot_differences_transfer('transfer-select-choice-rate-differences', colors=self.colors['group'], select=True)

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

    def raincloud_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.75, colors: list = []) -> None:
            
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
            colors : list
                The colors to use for the violin plot bodies, if not provided, will use a default color scheme

            Returns
            -------
            None
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

    def bar_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.75, colors: list = []) -> None:
            
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
            colors : list
                The colors to use for the bar plot, if not provided, will use a default color scheme

            Returns
            -------
            None
            """

            #Set index name
            data.index.name = 'code'
            if not isinstance(data, pd.DataFrame):
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

    def plot_combined_learning_and_transfer(self, save_name: str, learning_name: str, transfer_name: str) -> None:
        
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

        Returns (External)
        ------------------
        Image: PNG
            A combined image of the learning and transfer plots.
        """

        path = f'AL/plots/{self.split_by_group}'
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
        combined_path = f'AL/plots/{self.split_by_group}/{save_name}.png'
        combined_img.save(combined_path)

    def plot_learning_curves(self,
                             save_name: str,
                             rolling_mean: int = None, 
                             metric: str = 'accuracy',
                             grouping: str = 'clinical',
                             alpha: float = 0.75,
                             colors: list = [],
                             binned_trial: bool = True,
                             subplot_title: str = None) -> None:

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
        alpha : float
            The transparency of the lines in the plot
        colors : list
            The colors to use for the lines in the plot, if not provided, will use a default color scheme
        binned_trial : bool
            Whether to use binned trial numbers (Early, Mid-Early, Mid-Late, Late) or trial numbers (1-24)
        subplot_title : str
            The title to add to the subplot, if None, no title will be added

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
            if binned_trial:
                group_data = group_data[['participant_id', 'binned_trial', contexts_code, metric]]
                group_data = group_data.groupby(['participant_id', 'binned_trial', contexts_code]).mean().reset_index()
            else:
                group_data = group_data[['participant_id', 'trial_number', contexts_code, metric]]
                group_data = group_data.groupby(['participant_id', 'trial_number', contexts_code]).mean().reset_index()
                group_data['binned_trial'] = pd.Categorical(group_data['binned_trial'], categories=['Early', 'Mid-Early', 'Mid-Late', 'Late'], ordered=True)

            #Determine information of interest
            trial_index_name = 'binned_trial' if binned_trial else 'trial_number'
            for context_index, context in enumerate(contexts):
                context_data = group_data[group_data[contexts_code] == context]
                mean_accuracy = context_data.groupby(trial_index_name)[metric].mean()
                CIs = context_data.groupby(trial_index_name)[metric].sem()*t_score
                if rolling_mean is not None and binned_trial == False:
                    mean_accuracy = mean_accuracy.rolling(rolling_mean, min_periods=1, center=True).mean()
                
                if binned_trial:
                    mean_accuracy = mean_accuracy.reindex(['Early', 'Mid-Early', 'Mid-Late', 'Late'])
                    CIs = CIs.reindex(['Early', 'Mid-Early', 'Mid-Late', 'Late'])
                    ax[i].fill_between(np.arange(4), mean_accuracy - CIs, mean_accuracy + CIs, alpha=0.1, color=colors[context_index], edgecolor='none')
                    ax[i].plot(np.arange(4), mean_accuracy, color=colors[context_index], label=context.title(), linewidth=3, alpha=0.25)
                    ax[i].scatter(np.arange(4), mean_accuracy, color=colors[context_index], s=10, alpha=alpha)
                else:
                    ax[i].fill_between(mean_accuracy.index, mean_accuracy - CIs, mean_accuracy + CIs, alpha=0.1, color=colors[context_index], edgecolor='none')
                    ax[i].plot(mean_accuracy, color=colors[context_index], label=context.title(), linewidth=3, alpha=alpha)

            if metric == 'accuracy':
                ax[i].set_ylim(60, 100) if binned_trial else ax[i].set_ylim(40, 100)
            ax[i].set_title(f'{group.title()}')
            x_label = '' if binned_trial else 'Trial Number'
            ax[i].set_xlabel(x_label)
            ax[i].set_ylabel(f'{metric.capitalize()} (%)' if metric != 'rt' else 'Reaction Time (ms)')
            legend_loc = 'lower right' if metric != 'rt' else 'upper right'
            ax[i].legend(loc=legend_loc, frameon=False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].tick_params(axis='both')   

            if binned_trial:
                ax[i].set_xticks(np.arange(0, 4), ['Early', 'Mid-Early', 'Mid-Late', 'Late'], rotation=45)
                ax[i].set_xlim(-0.5, 3.5)
            else:
                ax[i].set_xticks(np.arange(0, 25, 4))

        #Add subplot labels
        if subplot_title:
            ax[0].annotate(subplot_title,
                            xy=(-.25, 1.15),
                            xytext=(0, 0),
                            xycoords='axes fraction',
                            textcoords='offset points',
                            ha='left',
                            va='top',
                            fontweight='bold')

        #Save the plot
        plt.tight_layout()
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def plot_transfer_data(self, save_name: str, colors: list, plot_type: str = 'raincloud', group_labels: bool = True, subplot_title: str = None) -> None:

        """
        Create raincloud plots of the data

        Parameters
        ----------
        save_name : str
            The name to save the plot as
        colors : list
            The colors to use for the plots, if not provided, will use a default color scheme
        plot_type : str
            The type of plot to create, either 'raincloud' or 'bar'
        group_labels : bool
            Whether to use group labels in the plot titles
        subplot_title : str
            The title to add to the subplot, if None, no title will be added

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
            #x_labels = ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel']
            x_labels = ['HR', 'LR', 'LP', 'HP', 'N']

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
        if subplot_title:
            ax[0].annotate(subplot_title,
                            xy=(-.25, 1.15),
                            xytext=(0, 0),
                            xycoords='axes fraction',
                            textcoords='offset points',
                            ha='left',
                            va='top',
                            fontweight='bold')

        plt.tight_layout()
        save_name = f'{save_name}_supplemental' if plot_type == 'raincloud' else save_name
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def plot_differences_transfer(self, save_name: str, colors: list, select: bool = False) -> None:

        """
        Create a raincloud plot of the difference in choice rates between low reward and low punish symbols for each group code

        Parameters
        ----------
        save_name : str
            The name to save the plot as
        colors : list
            The colors to use for the plot, if not provided, will use a default color scheme

        Returns (External)
        ------------------
        Image: PNG
            A plot of the raincloud plot
        """

        if not select:
            data = self.choice_rate
            data = data.reset_index()
            data = data[data['symbol'].isin(['High Reward', 'Low Reward', 'Low Punish'])]
            data = data.pivot(index=['participant_id', self.group_code], columns='symbol', values='choice_rate')
            data.reset_index(inplace=True)
            data['lr_lp'] = data.apply(lambda x: x['Low Reward'] - x['Low Punish'], axis=1)
            data['hr_lp'] = data.apply(lambda x: x['High Reward'] - x['Low Punish'], axis=1)
        else:
            hr_lp_data = self.select_choice_rate['High Reward'].reset_index()
            hr_lp_data = hr_lp_data[hr_lp_data['symbol'] == 'Low Punish']
            hr_lp_data['choice_rate'] = hr_lp_data['choice_rate'] - (100-hr_lp_data['choice_rate']) #The low punish is the reciprocal of the high reward, so we subtract it from 100 to get the choice rate for the high reward

            lr_lp_data = self.select_choice_rate['Low Reward'].reset_index()
            lr_lp_data = lr_lp_data[lr_lp_data['symbol'] == 'Low Punish']
            lr_lp_data['choice_rate'] = lr_lp_data['choice_rate'] - (100-lr_lp_data['choice_rate'])

            data = pd.merge(hr_lp_data[[self.group_code, 'participant_id', 'choice_rate']],
                            lr_lp_data[[self.group_code, 'participant_id', 'choice_rate']],
                            on=['participant_id', self.group_code], suffixes=('_hr', '_lr'))
            data = data.rename(columns={'choice_rate_hr': 'hr_lp', 'choice_rate_lr': 'lr_lp'})
        data[self.group_code] = pd.Categorical(data[self.group_code], categories=self.group_labels, ordered=True)

        #Create a raincloud plot the difference for each group code
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for sub in range(2):
            _, t_scores = self.compute_n_and_t(data, self.group_code)
            diff_label = 'hr_lp' if sub == 0 else 'lr_lp'
            metric_data = data.copy().set_index(self.group_code)[diff_label].astype(float)
            metric_data.index = pd.CategoricalIndex(metric_data.index, categories=self.group_labels, ordered=True)
            metric_data = metric_data.sort_index()
            self.bar_plot(data=metric_data, ax=ax[sub], t_scores=t_scores, colors=colors)

            ax[sub].set_xticks([1, 2, 3], ['No\nPain', 'Acute\nPain', 'Chronic\nPain'])
            ax[sub].set_xlabel('')
            y_label = 'Choice Rate: HR - LP (%)' if sub == 0 else 'Choice Rate: LR - LP (%)'
            ax[sub].set_ylabel(y_label)
            ax[sub].axhline(y=0, color='darkgrey', linestyle='--')
            ax[sub].spines['top'].set_visible(False)
            ax[sub].spines['right'].set_visible(False)
            ylim = [0, 50] if sub == 0 else [-10, 20]
            ax[sub].set_ylim(ylim)
            ax[sub].tick_params(axis='both')
            plt.subplots_adjust(left=0.2)
        
        plt.tight_layout()
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.png')
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
        x_ids = ['HR', 'LR', 'LP', 'HP', 'N']
        #x_labels = ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel']
        x_reference = {'High Reward': 'HR', 'Low Reward': 'LR', 'Low Punish': 'LP', 'High Punish': 'HP', 'Novel': 'N'}
        x_labels = ['HR', 'LR', 'LP', 'HP', 'N']
        plot_labels = self.group_labels
        
        #Create a bar plot of the choice rate for each symbol
        fig, ax = plt.subplots(5, 3, figsize=(15, 20))
        fig.subplots_adjust(hspace=0.35)
        for symbol_index, symbol in enumerate(['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel']):
            data = self.select_choice_rate[symbol]
            data = data.reset_index()
            data['symbol'] = data['symbol'].replace({'Novel': 0, 'High Reward': 4, 'Low Reward': 3, 'Low Punish': 2, 'High Punish': 1})
            data = data.set_index([self.group_code, 'participant_id', 'symbol'])
            symbol_x_ids = [x_id for x_id in x_ids if x_id != x_reference[symbol]]
            symbol_x_labels = [x_label for x_label in x_labels if x_label.replace('\n',' ') != x_reference[symbol]]
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
        plt.savefig(f'AL/plots/{self.split_by_group}/selected_{save_name}.png')
        plt.savefig(f'AL/plots/{self.split_by_group}/selected_{save_name}.svg', format='svg')

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
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

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
            self.raincloud_plot(data=metric_scores, ax=ax[metric_index], t_scores=t_scores, alpha=1, colors=colors)

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

        plt.subplots_adjust(bottom=0.2)

        #Save the plot
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.png')
        plt.savefig(f'AL/plots/{self.split_by_group}/{save_name}.svg', format='svg')

        #Close figure
        plt.close()
            
    def plot_model_parameters_by_pain(self, fit_data: pd.DataFrame, parameter_names: list, pain_names: list) -> None:
        
        """
        Plot model parameters against pain metrics for different pain groups.

        Parameters
        ----------
        fit_data : DataFrame
            DataFrame containing model parameters and pain metrics.
        parameter_names : list
            List of model parameter names to plot.
        pain_names : list
            List of pain metric names to plot against model parameters.

        Returns (External)
        -------
        Image: PNG
            A plot of model parameters against pain metrics for different pain groups.
        """

        fig, axes = plt.subplots(nrows=len(parameter_names), ncols=len(pain_names), figsize=(len(pain_names)*3, len(parameter_names)*3))
        colours = self.colors['group']
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

        plt.savefig(f'AL/plots/model_parameter_by_pain.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'AL/plots/model_parameter_by_pain.svg', format='svg', bbox_inches='tight')

        #Close figure
        plt.close(fig)

    def plot_model_parameters_by_pain_split(self, fit_data: pd.DataFrame, parameter_names: list, pain_names: list) -> None:

        """
        Plot model parameters against pain metrics for different pain groups, split by pain group.

        Parameters
        ----------
        fit_data : DataFrame
            DataFrame containing model parameters and pain metrics.
        parameter_names : list
            List of model parameter names to plot.
        pain_names : list
            List of pain metric names to plot against model parameters.

        Returns (External)
        -------
        Image: PNG
            A plot of model parameters against pain metrics for different pain groups, split by pain group.
        """

        cols = self.colors['group']
        colours = {'no pain': cols[0], 'chronic pain': cols[2], 'acute pain': cols[1]}
        
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
                                    alpha=0.8, color=colours[group], s=14)
                    
                    slope, intercept, _, _, _ = stats.linregress(correlation_data[pain_metric], 
                                                                correlation_data[parameter])
                    axes[i, j].plot(correlation_data[pain_metric], 
                                    slope * correlation_data[pain_metric] + intercept, 
                                    color=colours[group], linewidth=1.5)
                    
                    if i == len(parameter_names) - 1:
                        axes[i, j].set_xlabel(pain_metric.title(), fontsize=18)
                    else:
                        axes[i, j].set_xlabel('')
                    if j == 0:
                        y_label = parameter.replace('_', ' ').replace('lr', 'learning rate').replace(' learning', '\nlearning').replace('weighing ', 'weighting\n')
                        axes[i, j].set_ylabel(y_label.title(), fontsize=18)
                    
                    axes[i, j].set_xlim(np.floor(x_min), np.ceil(x_max))
                    axes[i, j].set_ylim(np.floor(y_min), np.ceil(y_max))
                    axes[i, j].tick_params(axis='both', which='major', labelsize=12)
                    axes[i, j].tick_params(axis='both', which='minor', labelsize=12)
                    axes[i, j].set_xticks(np.arange(0, 11, 2))
                    axes[i, j].set_xticklabels(np.arange(0, 11, 2), fontsize=12)
                    axes[i, j].spines['top'].set_visible(False)
                    axes[i, j].spines['right'].set_visible(False)
            
            plt.savefig(f'AL/plots/model_parameter_by_pain_{group.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'AL/plots/model_parameter_by_pain_{group.replace(" ", "_")}.svg', format='svg', bbox_inches='tight')

            #Close figure
            plt.close(fig)

    def plot_model_fits(self, fits: dict) -> None:

        """
        Plot the model fits for each group.

        Parameters
        ----------
        fits : dict
            Dictionary containing model fits for each group, where keys are fit names and values are DataFrames.

        Returns (External)
        ------------------
        Image: PNG
            A plot of the model fits for each group.
        """

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
                          alpha=0.75,
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
            plt.savefig(f'AL/modelling/group_{fit}.png')
            plt.close()

    def plot_param_duration(self, parameter_names, fit_data):
        parameter_names = list(parameter_names)
        duration_levels = fit_data['pain_duration'].unique()
        plot_data = fit_data[parameter_names+['pain_duration']].copy()

        significant_parameters = []
        nonsignificant_parameters = []
        for parameter in parameter_names:
            _, pval = stats.spearmanr(plot_data[parameter], plot_data['pain_duration'])
            if pval < 0.05:
                significant_parameters.append(parameter)
            else:
                nonsignificant_parameters.append(parameter)
        
        fig, ax = plt.subplots(nrows=1, ncols=len(parameter_names), figsize=(5*(len(parameter_names)), 5))
        for pi, parameter in enumerate(parameter_names):
            param_data = plot_data[[parameter, 'pain_duration']]
            param_data.set_index('pain_duration', inplace=True)
            _, t_scores = self.compute_n_and_t(param_data, 'pain_duration')

            self.bar_plot(param_data, ax[pi], t_scores=t_scores, colors=self.colors['group'][0])

            #Set the x-ticks and labels
            duration_key = {
                0: 'I am not in pain', 1: '< 2 weeks', 2: '2-4 weeks', 3: '1 – 3 months',
                4: '3 – 6 months', 5: '6 – 12 months', 6: '1 – 5 years', 7: '> 5 years', 8: '> 10 years'
            }
            duration_key = {k: v for k, v in duration_key.items() if k in duration_levels}
            ax[pi].set_xticks(np.arange(1, len(duration_key)+1))
            ax[pi].set_xticklabels(list(duration_key.values()), rotation=45, ha='right', va='top', fontsize=12)
            ax[pi].set_xlabel('')
            ax[pi].set_ylabel(parameter.replace('_', ' ').replace('lr', 'learning rate').replace(' learning', '\nlearning').title())
            ax[pi].spines['top'].set_visible(False)
            ax[pi].spines['right'].set_visible(False)
            ax[pi].tick_params(axis='both')
            
            plt.subplots_adjust(bottom=0.3)
            plt.tight_layout()

        #Save the plot
        plt.savefig(f'AL/plots/parameter_duration_corr.png')
        plt.close()

        if significant_parameters:
            fig, ax = plt.subplots(nrows=1, ncols=len(significant_parameters), figsize=(5 * len(significant_parameters), 5))
            if len(significant_parameters) == 1:
                ax = [ax]

            for pi, parameter in enumerate(significant_parameters):
                param_data = plot_data[[parameter, 'pain_duration']].copy()
                param_data.set_index('pain_duration', inplace=True)
                _, t_scores = self.compute_n_and_t(param_data, 'pain_duration')

                self.bar_plot(param_data, ax[pi], t_scores=t_scores, colors=self.colors['group'][0])

                duration_key = {
                    0: 'I am not in pain', 1: '< 2 weeks', 2: '2-4 weeks', 3: '1 – 3 months',
                    4: '3 – 6 months', 5: '6 – 12 months', 6: '1 – 5 years', 7: '> 5 years', 8: '> 10 years'
                }
                duration_key = {k: v for k, v in duration_key.items() if k in duration_levels}
                ax[pi].set_xticks(np.arange(1, len(duration_key) + 1))
                ax[pi].set_xticklabels(list(duration_key.values()), rotation=45, ha='right', va='top', fontsize=12)
                ax[pi].set_xlabel('')
                ax[pi].set_ylabel(parameter.replace('_', ' ').replace('lr', 'learning rate').replace(' learning', '\nlearning').title())
                ax[pi].spines['top'].set_visible(False)
                ax[pi].spines['right'].set_visible(False)
                ax[pi].tick_params(axis='both')

            plt.subplots_adjust(bottom=0.3)
            plt.tight_layout()
            plt.savefig(f'AL/plots/parameter_duration_corr-significant.png')
            plt.close()

        # === Plot non-significant parameters ===
        if nonsignificant_parameters:
            fig, ax = plt.subplots(nrows=1, ncols=len(nonsignificant_parameters), figsize=(5 * len(nonsignificant_parameters), 5))
            if len(nonsignificant_parameters) == 1:
                ax = [ax]

            for pi, parameter in enumerate(nonsignificant_parameters):
                param_data = plot_data[[parameter, 'pain_duration']].copy()
                param_data.set_index('pain_duration', inplace=True)
                _, t_scores = self.compute_n_and_t(param_data, 'pain_duration')

                self.bar_plot(param_data, ax[pi], t_scores=t_scores, colors=self.colors['group'][0])

                duration_key = {
                    0: 'I am not in pain', 1: '< 2 weeks', 2: '2-4 weeks', 3: '1 – 3 months',
                    4: '3 – 6 months', 5: '6 – 12 months', 6: '1 – 5 years', 7: '> 5 years', 8: '> 10 years'
                }
                duration_key = {k: v for k, v in duration_key.items() if k in duration_levels}
                ax[pi].set_xticks(np.arange(1, len(duration_key) + 1))
                ax[pi].set_xticklabels(list(duration_key.values()), rotation=45, ha='right', va='top', fontsize=12)
                ax[pi].set_xlabel('')
                ax[pi].set_ylabel(parameter.replace('_', ' ').replace('lr', 'learning rate').replace(' learning', '\nlearning').title())
                ax[pi].spines['top'].set_visible(False)
                ax[pi].spines['right'].set_visible(False)
                ax[pi].tick_params(axis='both')

            plt.suptitle("Non-Significant Parameters", fontsize=16, x=0.01, ha='left')
            plt.subplots_adjust(bottom=0.3)
            plt.tight_layout()
            plt.savefig(f'AL/plots/parameter_duration_corr-nonsignificant.png')
            plt.close()
                

