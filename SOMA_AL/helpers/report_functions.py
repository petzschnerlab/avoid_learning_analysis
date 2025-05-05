import os
import warnings
import pandas as pd
import dataframe_image as dfi
import numpy as np
import pickle
import scipy.stats as stats
import copy

from markdown_pdf import Section
from helpers.statistics import Statistics
from helpers.processing import Processing
from helpers.plotting import Plotting


class ReportFunctions:
    """
    This class contains functions for generating reports.
    """

    def save_report(self) -> None:

        """
        Saves the report as a pdf

        Returns (External)
        ------------------
        Report: PDF
            The PEAC report
        """

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

    def print_planned_statistics(self, comparison: str, model_summary: pd.DataFrame) -> str:

        """
        Prints the planned comparison statistics
        
        Parameters
        ----------
        comparison : str
            The comparisons being made
        model_summary : pd.DataFrame
            The summary statistics for the comparison

        Returns
        -------
        subsection : str
            The subsection for the planned comparison
        """

        if '~' in comparison:
            comparisons = comparison.split(' vs ')
            comparison = f"{comparisons[0].split('~')[0]} vs {comparisons[1].split('~')[0]}: {comparisons[0].split('~')[1]}"
        significance = '\*' if (model_summary['p_value'].values < 0.05) else ''
        df = round(float(model_summary['df'].values[0]))
        test_value = model_summary['t_value'].values[0].round(2)
        p = model_summary['p_value'].values[0].round(4) if (model_summary['p_value'] >= 0.0001).values else '<0.0001'
        p = f'= {p}' if p != '<0.0001' else p
        d = model_summary['cohens_d'].values[0].round(2)

        if self.hide_stats:
            significance = ''
            df = '__'
            test_value = 'hidden'
            p = '= hidden'
            d = 'hidden'

        subsection = f"\n\n&nbsp;&nbsp;&nbsp;&nbsp;**{comparison.title()}**"
        t_type = '<sub>Welch</sub>' if model_summary['homogeneity_assumption'].values[0] == 'violated' else ''
        subsection += f"\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*t{t_type}*(*{df}*) = {test_value}, *p* {p}{significance}, *d* = {d}."

        return subsection

    #Formatting functions
    def add_data_pdf(self, content: list, toc: bool = True, center: bool = False) -> None:

        """
        Adds the content to the pdf
        
        Parameters
        ----------
        content : list
            The content to add to the pdf
        toc : bool
            Whether to add a table of contents
        center : bool
            Whether to center the content

        Returns (External)
        ------------------ 
        Report: PDF
            A section of the PEAC report
        """

        #Formatting
        user_css = 'h4 {text-align:center;}' if center else None
        section = Section(' \n '.join(content), toc=toc)
        self.pdf.add_section(section, user_css=user_css)

    def table_to_png(self, table: pd.DataFrame, save_name: str = "SOMA_AL/plots/tables/Table.png") -> None:

        """
        Converts a table to a png

        Parameters
        ----------
        table : pd.DataFrame
            The table to convert
        save_name : str
            The name to save the table as

        Returns (External)
        ------------------
        Image: PNG
            The table as a png
        """
        
        #Format titles as titles
        for i in range(len(table)):
            table.index.values[i] = table.index.values[i].title()
        table.columns = table.columns.str.title()
        table.columns.name = None   

        #Format the table
        table = table.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '10pt'), 
                                                                           ('text-align', 'center'), 
                                                                           ('background-color', '#FFFFFF')]},

                                                {'selector': 'td', 'props': [('font-size', '10pt'), 
                                                                             ('text-align', 'center'), 
                                                                             ('background-color', '#FFFFFF')]},

                                                {'selector': '', 'props': [('border-top', '1px solid black'), 
                                                                           ('border-bottom', '1px solid black'),
                                                                           ('border-left', '1px solid white'),
                                                                           ('border-right', '1px solid white')]},])
        
        #Save the table as a png
        dfi.export(table, save_name, table_conversion="selenium", max_rows=-1)
    
    #Data retrieval functions   
    def add_figure_caption(self, text: str) -> str:

        """
        Adds a figure mumner to the caption

        Parameters
        ----------
        text : str
            The text to add to the caption

        Returns
        -------
        section_text : str
            The section text with the figure number
        """

        section_text = f'**Figure {self.figure_count}.** {text}'
        self.figure_count += 1

        return section_text
    
    def add_table_caption(self, text: str) -> str:

        """
        Adds a table number to the caption

        Parameters
        ----------
        text : str
            The text to add to the caption

        Returns
        -------
        section_text : str
            The section text with the table number
        """

        section_text = f'**Table {self.table_count}.** {text}'
        self.table_count += 1
        
        return section_text

    def get_caption(self, target: str, target_type: str = 'figure') -> str:

        """
        Gets the caption for the target

        Parameters
        ----------
        target : str
            The target to get the caption for
        target_type : str
            The type of target

        Returns
        -------
        caption : str
            The caption for the target
        """
        
        caption = ''
        match target:
            case 'demo-scores':
                target_type = 'table'
                caption = 'Demographic information for each group.'
                if self.split_by_group == 'pain':
                    caption += ' Group differences reflect which groups are significantly different from the no pain group in planned follow-up tests.'

            case 'demo-clinical-scores':
                depression_caption = ' and depression' if self.split_by_group == 'depression' else ''
                caption = f'''Pain{depression_caption} metrics for each group.
                Boxplots show the mean and 95\% confidence intervals of the corresponding metric for each group.
                Half-violin plots show the distribution of the scores of the corresponding metric for each group.
                Scatter points show the scores of the corresponding metric for each participant within each group.'''

            case 'learning-accuracy-by-group':
                caption = 'Behavioral performance across learning trials for the reward and punishment contexts for each group.'
                if self.rolling_mean is not None:
                    caption += f' For visualization, the accuracy is smoothed using a rolling mean of {self.rolling_mean} trials.'
                caption += ' Shaded regions represent 95\% confidence intervals.'

            case 'learning-accuracy':
                caption = """Averaged behavioral performance during learning for the reward and punishment contexts for each group.
                Boxplots show the mean and 95\% confidence intervals of the accuracy for each context across participants within each group.
                Half-violin plots show the distribution of accuracy for each context across participants within each group.
                Scatter points show the averaged accuracy for each participant within each context."""

            case 'learning-accuracy-diff':
                caption = """Averaged difference (Reward - Punish) of behavioral performance during learning for each group.
                Boxplots show the mean and 95\% confidence intervals of the accuracy for each context across participants within each group.
                Half-violin plots show the distribution of accuracy for each context across participants within each group.
                Scatter points show the averaged accuracy for each participant within each symbol type."""

            case 'learning-rt':
                caption = """Averaged behavioral performance during learning for the reward and punishment contexts for each group.
                Boxplots show the mean and 95\% confidence intervals of the reaction times for each context across participants within each group.
                Half-violin plots show the distribution of reaction times for each context across participants within each group.
                Scatter points show the averaged reaction times for each participant within each context."""

            case 'learning-rt-diff':
                caption = """Averaged difference (Reward - Punish) of behavioral performance during learning for each group.
                Boxplots show the mean and 95\% confidence intervals of the reaction times for each context across participants within each group.
                Half-violin plots show the distribution of reaction times for each context across participants within each group.
                Scatter points show the averaged reaction times for each participant within each symbol type."""

            case 'learning-accuracy-by-context':
                caption = 'Behavioral performance across learning trials for each group for each context.'
                if self.rolling_mean is not None:
                    caption += f' For visualization, the accuracy is smoothed using a rolling mean of {self.rolling_mean} trials.'
                caption += ' Shaded regions represent 95\% confidence intervals.'

            case 'learning-rt-by-group':
                caption = 'Reaction times across learning trials for the reward and punishment contexts for each group.'
                if self.rolling_mean is not None:
                    caption += f' For visualization, the reaction time is smoothed using a rolling mean of {self.rolling_mean} trials.'
                caption += ' Shaded regions represent 95\% confidence intervals.'

            case 'learning-rt-by-context':
                caption = 'Reaction times across learning trials for each group for each context.'
                if self.rolling_mean is not None:
                    caption += f' For visualization, the reaction time is smoothed using a rolling mean of {self.rolling_mean} trials.'
                caption += ' Shaded regions represent 95\% confidence intervals.'

            case 'transfer-choice-rate':
                caption = """Choice rate for each symbol during transfer trials for each group.
                Choice rate is computed as the percentage of times a symbol was chosen given the number of times it was presented.
                Boxplots show the mean and 95\% confidence intervals of the choice rate for each symbol type across participants within each group.
                Half-violin plots show the distribution of choice rates for each symbol type across participants within each group.
                Scatter points show the averaged choice rate for each participant within each symbol type."""

            case 'transfer-choice-rate-neutral':
                caption = """Choice rates for cases where the low reward was compared to the low punishment symbols in the transfer trials. 
                Choice rates represent the percentage of times the low reward was chosen, thus 50\% indicates equal choice rates for both symbols,
                while greater than 50% indicates a preference for the low reward symbol. 
                Boxplots show the mean and 95\% confidence intervals of the choice rate for each group."""

            case 'transfer-rt':
                caption = """Reaction times for each symbol during transfer trials for each group.
                Boxplots show the mean and 95\% confidence intervals of the reaction times for each symbol type across participants within each group.
                Half-violin plots show the distribution of reaction times for each symbol type across participants within each group.
                Scatter points show the averaged reaction time for each participant within each symbol type."""

            case 'transfer-rt-neutral':
                caption = """Reaction times for cases where the low reward was compared to the low punishment symbols in the transfer trials. 
                Boxplots show the mean and 95\% confidence intervals of the reaction times for each group."""

            case 'transfer-valence-bias':
                caption = """Valence bias for each group. Valence bias is computed as the difference in choice rate differences in each context, 
                specifically (High Reward - Low Reward) - (Low Punish - High Punish). 
                Positive biases indicate that there was a larger of choice rates within the reward context, 
                and negative biases indicate that there was a larger difference in choice rates within the punishment context.
                Boxplots show the mean and 95\% confidence intervals of the bias for each group.
                Half-violin plots show the distribution of biases across participants within each group.
                Scatter points show the bias for each participant within each group.
                """ 

            case 'model-AIC':
                caption = """Akaike Information Criterion (AIC) metrics for each group and the full dataset. 
                Lower values indicate better model fit. The final column highlights the best-performing model for each group.
                """

            case 'model-BIC':
                caption = """Bayesian Information Criterion (BIC) metrics for each group and the full dataset. 
                Lower values indicate better model fit. The final column highlights the best-performing model for each group.
                """

            case 'model-AIC-percentages':
                caption = """Percentage of times a model was selected as the best model via Akaike Information Criterion (AIC) metrics across all participants within and across each group.
                """

            case 'model-BIC-percentages':
                caption = """Percentage of times a model was selected as the best model via Bayesian Information Criterion (BIC) metrics across all participants within and across each group.
                """

            case 'fit-by-runs':
                caption = f"""Negative log-likelihood (NLL) as a factor of the number of runs for each model. 
                For each run, the starting parameters are randomly determined. 
                Dashed red line represents the median run that achieved the lowest NLL across all models.
                """

            case 'model-recovery':
                caption = """Confusion matrix showing the model recovery for each model.
                Each model was fit to recover generated data derived from each model.
                Fit values represent BIC fits, thus lower values indicate better model recovery.
                The diagonal values represent the model's ability to recover itself, 
                while off-diagonal values represent the model's ability to recover other models.
                """

            case 'model-behaviour':
                caption = f"""
                """
                caption = f"""Best Model, {self.best_model_label}, simulated behavioral data. 
                Top: Accuracy across learning trials for the reward and punishment contexts for each group. 
                """
                if self.rolling_mean is not None:
                    caption += f'For visualization, the accuracy is smoothed using a rolling mean of {self.rolling_mean} trials. '
                caption += 'Shaded regions represent 95\% confidence intervals. Dashed lines represent averaged empirical data. '
                caption += """Bottom: Choice rate for each symbol during transfer trials for each group. 
                Choice rate is computed as the percentage of times a symbol was chosen given the number of times it was presented.
                Bars indicate the averaged choice rate for each symbol type across participants within each group.
                Error bars represent the 95\% confidence intervals of the choice rate for each symbol type across participants within each group.
                Grey diamonds represent averaged empirical data.
                """

            case 'model-parameters':
                caption = f"""Best Model, {self.best_model_label}, simulated parameter fits for each group.
                Each parameter is shown as a boxplot with the mean and 95\% confidence intervals across participants within each group.
                Half-violin plots show the distribution of parameter fits across participants within each group.
                Scatter points show the parameter fits for each participant within each group.
                """
            
            case 'parameter-correlation':
                caption = f"""Correlations between the best model's, {self.best_model_label}, parameters and pain scores. 
                Scatter points show the parameter fits for each participant within each group, grey dashed lines represent the best fit line for each group.
                """
                
            case 'parameter-correlation-no-pain':
                caption = f"""Correlations between the best model's, {self.best_model_label}, parameters and pain scores for the No Pain group. 
                Scatter points show the parameter fits for each participant within each group, grey dashed lines represent the best fit line for each group.
                """

            case 'parameter-correlation-acute-pain':
                caption = f"""Correlations between the best model's, {self.best_model_label}, parameters and pain scores for the Acute Pain group. 
                Scatter points show the parameter fits for each participant within each group, grey dashed lines represent the best fit line for each group.
                """

            case 'parameter-correlation-chronic-pain':
                caption = f"""Correlations between the best model's, {self.best_model_label}, parameters and pain scores for the Chronic Pain group. 
                Scatter points show the parameter fits for each participant within each group, grey dashed lines represent the best fit line for each group.
                """
            
            case 'model-parameters-correlation-table':
                caption = f"""Correlation table between the best model's, {self.best_model_label}, parameters and pain scores. 
                Each cell represents the Pearson's r value (p-value) for the correlation between the parameter and pain score for all groups.
                """
            case 'model-parameters-correlation-table-no':
                caption = f"""Correlation table between the best model's, {self.best_model_label}, parameters and pain scores for the no pain group. 
                Each cell represents the Pearson's r value (p-value) for the correlation between the parameter and pain score.
                """

            case 'model-parameters-correlation-table-acute':
                caption = f"""Correlation table between the best model's, {self.best_model_label}, parameters and pain scores for the acute pain group.
                Each cell represents the Pearson's r value (p-value) for the correlation between the parameter and pain score.
                """

            case 'model-parameters-correlation-table-chronic':
                caption = f"""Correlation table between the best model's, {self.best_model_label}, parameters and pain scores for the chronic pain group.
                Each cell represents the Pearson's r value (p-value) for the correlation between the parameter and pain score.
                """

            case 'empirical-performance':
                caption = 'Empirical findings for the learning phase (top) and the transfer phase (bottom). Learning Phase (Top): Behavioral performance across learning trials for the reward and punishment contexts for each group.'
                if self.rolling_mean is not None:
                    caption += f' For visualization, the accuracy is smoothed using a rolling mean of {self.rolling_mean} trials.'
                caption += ' Shaded regions represent 95\% confidence intervals.'
                caption += """ Transfer Phase (Bottom): Choice rate for each symbol during transfer trials for each group.
                Choice rate is computed as the percentage of times a symbol was chosen given the number of times it was presented.
                Bar plots show the mean and 95\% confidence intervals of the choice rate for each symbol type across participants within each group."""

            case 'empirical-rt':
                caption = 'Empirical findings for the learning phase (top) and the transfer phase (bottom). Learning Phase (Top): Reaction times across learning trials for the reward and punishment contexts for each group.'
                if self.rolling_mean is not None:
                    caption += f' For visualization, the reaction time is smoothed using a rolling mean of {self.rolling_mean} trials.'
                caption += ' Shaded regions represent 95\% confidence intervals.'
                caption += """ Transfer Phase (Bottom): Reaction times for each symbol during transfer trials for each group.
                Bar plots show the mean and 95\% confidence intervals of the reaction times for each symbol type across participants within each group."""

        if 'correlation-plot' in target:
            caption = f"""Parameter recovery correlational plot for the {target.split('-')[0].title()} model, 
            comparing the true parameter values to the estimated parameter values.
            The dashed line represents the ideal 1:1 relationship.
            Correlational strengths are represented by the Pearson's r value, and presented in the title of each subplot.
            """
        
        if 'model-parameter-posthoc' in target:
            #Table of posthocs across groups for the given parameter
            caption = f"""Tukey Post-Hoc results for the {target.split('-')[0].title()} parameter across groups."""

        #Return caption
        if target_type == 'figure':
            caption = self.add_figure_caption(caption)
        else:
            caption = self.add_table_caption(caption)

        return caption
    
    def get_metadata(self, data: dict) -> tuple:
        
        """
        Gets the metadata for the data

        Parameters
        ----------
        data : dict
            The data to get the metadata for

        Returns
        -------
        formula : str
            The formula for the analysis
        outcome : str
            The outcome for the analysis
        fixed_effects : list
            The fixed effects for the analysis
        random_effects : list
            The random effects for the analysis
        sample_size : int
            The sample size for the analysis
        df_residual : int
            The residual degrees of freedom for the analysis
        test : str
            The test used for the analysis
        """

        formula = data['metadata']['formula']
        fixed_effects = data['metadata']['fixed_effects']
        random_effects = data['metadata']['random_effects']
        outcome = data['metadata']['outcome']
        sample_size = data['metadata']['sample_size']
        test = data['metadata']['test']
        df_residual = data['metadata']['df_residual']

        return formula, outcome, fixed_effects, random_effects, sample_size, df_residual, test
    
    def get_statistics(self, target: str) -> list[str]:

        """
        Gets the statistics for the target

        Parameters
        ----------
        target : str
            The target to get the statistics for

        Returns
        -------
        subsection : list
            The subsection for the statistics
        """

        self.data_legend = {'learning-accuracy': self.learning_accuracy_glmm,
                            'learning-rt': self.learning_rt_glmm,
                            'transfer-choice-rate': self.transfer_accuracy_glmm,
                            'transfer-rt': self.transfer_rt_glmm,
                            'demographics-and-clinical-scores': self.demo_clinical,
                            }
        
        if self.load_models:
            self.model_legend = {'learning-model-behaviour-accuracy': self.model_learning_accuracy_glmm,
                                 'transfer-model-behaviour-choice-rate': self.learning_accuracy_glmm,
                                'model-parameters': self.learning_accuracy_glmm
            }
            self.data_legend.update(self.model_legend)

        data = self.data_legend[target]
        formula, outcome, fixed, random, sample_size, df_residual, test = self.get_metadata(data)
        outcome = 'Parameters' if target == 'model-parameters' else outcome
        test = 'χ<sup>2</sup>' if test == 'Chisq' else test
        phase = target.split('-')[0]
        summary = data['model_summary']

        if self.split_by_group == 'pain':
            self.data_planned_legend = {'learning-accuracy-by-group': self.learning_accuracy_planned_group,
                                        'learning-rt-by-group': self.learning_rt_planned_group,
                                        'transfer-choice-rate-by-group': self.transfer_accuracy_planned_group,
                                        'transfer-rt-by-group': self.transfer_rt_planned_group,

                                        'transfer-choice-rate-by-context': self.transfer_accuracy_planned_context,
                                        'transfer-rt-by-context': self.transfer_rt_planned_context,

                                        'learning-accuracy-by-interaction': self.learning_accuracy_planned_interaction,
                                        'learning-rt-by-interaction': self.learning_rt_planned_interaction,
                                        'transfer-choice-rate-by-interaction': self.transfer_accuracy_planned_interaction,
                                        'transfer-rt-by-interaction': self.transfer_rt_planned_interaction,

                                        'demographics-and-clinical-scores': self.demo_clinical_planned,
                                        }
            
            if self.load_models:
                self.model_planned_legend = {'learning-model-behaviour-accuracy-by-group': self.model_learning_accuracy_planned_group,
                                            'learning-model-behaviour-accuracy-by-interaction': self.model_learning_accuracy_planned_interaction,
                                            'transfer-model-behaviour-choice-rate-by-group': self.model_transfer_choice_rate_planned_group,
                                            'transfer-model-behaviour-choice-rate-by-context': self.model_transfer_choice_rate_planned_context,
                                            'transfer-model-behaviour-choice-rate-by-interaction': self.model_transfer_accuracy_planned_interaction,
                                            'model-parameters-by-group': self.model_parameters_planned_group}
                self.data_planned_legend.update(self.model_planned_legend)
                
        else:
            self.data_planned_legend = {'transfer-choice-rate-by-context': self.transfer_accuracy_planned_context,
                                        'transfer-rt-by-context': self.transfer_rt_planned_context,

                                        'learning-accuracy-by-interaction': self.learning_accuracy_planned_interaction,
                                        'learning-rt-by-interaction': self.learning_rt_planned_interaction,
                                        'transfer-choice-rate-by-interaction': self.transfer_accuracy_planned_interaction,
                                        'transfer-rt-by-interaction': self.transfer_rt_planned_interaction}
            
        subsection = f'**{target.replace("-", " ").title()} Statistics**\n\n'
        if target == 'model-parameters':
            subsection += f"""Parameters for the best model, {self.best_model_label}, were modelled a linear mixed effects model with the following formula: *{formula.replace('*',':').replace('group_code','group').replace('symbol_name','context')}*.
            """
        elif data['metadata']['outcome'] == 'metric':
            outcomes = ', '.join(data['model_summary']['factor'].unique())
            subsection += f"""{outcomes.capitalize()} were modelled using linear regression with the following formula: *{formula}*."""
        else:
            subsection += f"""{outcome.capitalize()} in the {phase} phase was modelled using a linear mixed effects model with the following formula: *{formula.replace('*', ':').replace('group_code','group').replace('symbol_name','context')}*, 
            where *{', '.join([f.replace('*',':').replace('group_code','group').replace('symbol_name','context') for f in fixed])}* are the fixed effects {f'and *{random}* is the random effect.' if random else '.'}"""
        
        if target != 'demographics-and-clinical-scores':
            subsection += ' Following each main and interaction finding from the linear model, we report planned comparison t-tests, corrected using a Welch\'s t-test when the assumption of homogeneity of variance was violated.'
            if self.hide_posthocs == False:
                subsection += (' Further, Tukey HSD post-hoc comparisons can be found in Appendix A.')

        #Iterate through summary and format each row into a sentence
        for i, factor in enumerate(summary['factor'].unique()):
            factor_data = summary[summary['factor']==factor]
            significance = '\*' if (factor_data['p_value'].values < 0.05) else ''
            df_1 = round(float(factor_data['df'].values[0])) #TODO: Check these - they are not the same as k-1 but probably bc of the method used
            df_2 = round(float(df_residual)) if test == 'F' else f'N={sample_size}'
            test_value = factor_data['test_value'].values.round(2)[0]
            p = factor_data['p_value'].values.round(3)[0] if (factor_data['p_value'] >= 0.001).values else '<0.001'
            p = f'= {p}' if p != '<0.001' else p

            if self.hide_stats:
                significance = ''
                df_1 = '__'
                df_2 = '__'
                test_value = 'hidden'
                p = ' = hidden'

            factor_named = factor.replace('*',':').replace('group_code','group').replace('symbol_name','context')
            subsection += f"\n\n**{factor_named.title()}{significance}:** *{test}*(*{df_1}, {df_2}*) = {test_value}, *p* {p}"

            #Add planned comparisons
            if target == 'model-parameters':
                planned_target = f'{target}-by-group'
            elif 'binned_trial' in factor or 'depression' == factor:
                planned_target = ''
            elif ':' in factor:
                planned_target = f'{target}-by-interaction'
            elif 'group' in factor:
                planned_target = f'{target}-by-group'
            else:
                planned_target = f'{target}-by-context'

            if planned_target in self.data_planned_legend.keys():
                planned_summary = self.data_planned_legend[planned_target]['model_summary']

                if 'factor' in planned_summary.columns:
                    factor_summary = planned_summary[planned_summary['factor']==factor]
                else:
                    factor_summary = planned_summary

                for comparison in factor_summary['comparison'].unique():
                    comparison_summary = factor_summary[factor_summary['comparison']==comparison]
                    subsection += self.print_planned_statistics(comparison, comparison_summary)

        return [subsection]
    
    #Content builders
    def insert_image(self, image_name: str, filename = None) -> list[str]:

        """
        Inserts an image for the report

        Parameters
        ----------
        image_name : str
            The name of the image

        Returns
        -------
        subsection : list
            The subsection for the report
        """

        filename = f'SOMA_AL/plots/{self.split_by_group}/{image_name}.png' if filename is None else filename
        subsection = [f'#### ![{image_name}]({filename})\n', 
                      f'{self.get_caption(image_name)}\n']
       
        return subsection
    
    def insert_table(self, table: pd.DataFrame, save_name: str, max_rows: int = None) -> list[str]:

        """
        Inserts a table for the report

        Parameters
        ----------
        table : pd.DataFrame
            The table to insert
        save_name : str
            The name to save the table as
        max_rows : int
            The maximum number of rows to display in each subtable

        Returns
        -------
        subsection : list
            The subsection for the report
        """

        #Set title
        subsection = [f'{self.get_caption(save_name, target_type="table")}']
        
        #Split into subtables
        if max_rows is not None:
            table = table.reset_index()
            n = len(table)
            subtables = [table[i:i+max_rows] for i in range(0, n, max_rows)]
            for i, subtable in enumerate(subtables):
                if self.load_posthocs == False:
                    self.table_to_png(subtable.set_index('factor'), save_name=f'SOMA_AL/plots/tables/{self.split_by_group}_{save_name}_{i}.png')
                subsection += [f'#### ![{save_name}_{i}](SOMA_AL/plots/tables/{self.split_by_group}_{save_name}_{i}.png)\n']
        else: #Print full table
            if self.load_posthocs == False:
                self.table_to_png(table, save_name=f'SOMA_AL/plots/tables/{self.split_by_group}_{save_name}.png')
            subsection += [f'#### ![{save_name}](SOMA_AL/plots/tables/{self.split_by_group}_{save_name}.png)\n']
       
        return subsection
    
    def insert_title_page(self) -> None:

        """
        Inserts the title page for the report

        Returns (External)
        ------------------
        Report: PDF
            Adds the title page to the SOMA report
        """

        section_text = [f'# PEAC Lab Report',
                        f'![PEAC_logo](SOMA_AL/media/PEAC_logo.png)']
        self.add_data_pdf(section_text)

    def insert_report_details(self) -> None:

        """
        Inserts the report details for the report

        Returns (External)
        ------------------
        Report: PDF
            Adds the report details to the SOMA report
        """

        section_text = [f'## PEAC Report Details',
                        f'**Generated by:** {self.author}\n',
                        f'**Date:** {str(pd.Timestamp.now()).split(" ")[0]}',
                        #Add a description of all of the kwargs
                        f'\n### Inputted Parameters']
        for key, value in self.kwargs.items():
            section_text += [f'**{key.replace("_", " ").title()}:** {value}\n']
        section_text.append(f'\n\n{"*Note: statistics are hidden, to reveal them, set hide_stats=False.*" if self.hide_stats else ""}')
        self.add_data_pdf(section_text)

    def insert_analysis_details(self) -> None:

        """
        Inserts the analysis details for the report

        Returns (External)
        ------------------
        Report: PDF
            Adds the analysis details to the PEAC report
        """

        section_text = [f'## Data Characteristics',
                        f'**File{"s" if len(self.file_name) > 1 else ""}:** {", ".join(self.file_name)}',
                        f'### Grouping',
                        f'**Split by Group:** {self.split_by_group.capitalize()}',
                        f'### Column Names',
                        f'{", ".join(self.data.columns)}',
                        f'### Data Dimensions',
                        f'**Rows:** {self.data.shape[0]}\n',
                        f'**Columns:** {self.data.shape[1]}\n',
                        f'**Number of Groups:** {len(self.group_labels)}\n',
                        f'**Number of Original Participants:** {self.participants_original}\n',
                        f'**Number of Participants Excluded (Pain Threshold): {self.number_pain_excluded}**\n',
                        f'**Number of Participants Excluded (Accuracy Threshold: {self.accuracy_threshold}%):** {self.number_accuracy_excluded}\n',
                        f'**Number of Participants Remaining:** {self.learning_data["participant_id"].nunique()}\n',
                        f'**Percentage of Trials Excluded (RT Threshold: < {self.RT_low_threshold}ms or > {self.RT_high_threshold}ms):** {self.trials_excluded_rt.round(2)}%\n']
        self.add_data_pdf(section_text)
    
    def insert_demographics_table(self) -> None:

        """
        Inserts the demographics table for the report

        Returns (Internal)
        ------------------
        self.demographics : pd.DataFrame
            The demographics table for the report
        """
        
        column_blanks = ['','','',''] if self.split_by_group == 'pain' else ['','','']
        demo_title = pd.DataFrame([column_blanks], columns=self.demographics_summary.columns, index=['Demographics'])
        pain_title = pd.DataFrame([column_blanks], columns=self.demographics_summary.columns, index=['Pain Scores'])
        depression_title = pd.DataFrame([column_blanks], columns=self.demographics_summary.columns, index=['Depression Scores'])
        blank_row = pd.DataFrame([column_blanks], columns=self.demographics_summary.columns, index=[''])
        self.demographics = pd.concat([blank_row,
                                        demo_title,
                                        self.demographics_summary, 
                                        blank_row,
                                        pain_title, 
                                        self.pain_summary], axis=0)
        
        if self.depression_summary is not None:
            self.demographics = pd.concat([self.demographics, 
                                           blank_row, 
                                           depression_title, 
                                           self.depression_summary], axis=0)
            
    def load_modelling_results(self, rscripts_path=None, load_stats=False):

        #Assign attributes
        self.rscripts_path = rscripts_path

        #Model Fits
        self.model_AIC = pd.read_csv('SOMA_AL/modelling/group_AIC.csv')
        self.model_BIC = pd.read_csv('SOMA_AL/modelling/group_BIC.csv')
        self.model_AIC_percentages = pd.read_csv('SOMA_AL/modelling/group_AIC_percentages.csv')
        self.model_BIC_percentages = pd.read_csv('SOMA_AL/modelling/group_BIC_percentages.csv')

        self.model_AIC.set_index('Unnamed: 0', inplace=True)
        self.model_BIC.set_index('Unnamed: 0', inplace=True)
        self.model_AIC_percentages.set_index('group', inplace=True)
        self.model_BIC_percentages.set_index('group', inplace=True)

        self.model_AIC.index.name = None
        self.model_BIC.index.name = None
        self.model_AIC_percentages.index.name = None
        self.model_BIC_percentages.index.name = None

        self.model_AIC = self.model_AIC.iloc[:,:-1].round().astype(int)
        self.model_BIC = self.model_BIC.iloc[:,:-1].round().astype(int)
        self.model_AIC_percentages = self.model_AIC_percentages.round(0).astype(int)
        self.model_BIC_percentages = self.model_BIC_percentages.round(0).astype(int)

        self.model_AIC['best_model'] = self.model_AIC.idxmin(axis=1)
        self.model_BIC['best_model'] = self.model_BIC.idxmin(axis=1)
        best_model_BIC = self.model_BIC['best_model'].values[-1]
        self.best_model = best_model_BIC
        self.best_model_label = best_model_BIC.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning').replace('best_model', 'Best Model')
        
        self.model_names = list(self.model_AIC.columns[:-1])
        self.model_AIC.columns = [col.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning').replace('best_model', 'Best Model') for col in self.model_AIC.columns]
        self.model_BIC.columns = [col.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning').replace('best_model', 'Best Model') for col in self.model_BIC.columns]
        self.model_AIC_percentages.columns = [col.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning').replace('best_model', 'Best Model') for col in self.model_AIC_percentages.columns]
        self.model_BIC_percentages.columns = [col.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning').replace('best_model', 'Best Model') for col in self.model_BIC_percentages.columns]

        #Load model simulation data
        self.model_accuracy = pd.read_csv(f'SOMA_AL/modelling/modelsimulation_accuracy_data.csv')
        self.model_choice_rate = pd.read_csv(f'SOMA_AL/modelling/modelsimulation_choice_data.csv')

        self.model_accuracy.rename(columns={'context': 'symbol_name', 'run': 'participant_id', 'group': self.group_code}, inplace=True)
        self.model_choice_rate.rename(columns={'group': self.group_code}, inplace=True)
        self.model_accuracy['symbol_name'].replace({'Loss Avoid': 'Punish'}, inplace=True)

        self.model_accuracy = self.model_accuracy[self.model_accuracy['model'] == self.best_model]
        self.model_choice_rate = self.model_choice_rate[self.model_choice_rate['model'] == self.best_model]

        self.model_accuracy['binned_trial'] = np.ceil(self.model_accuracy['trial_total'] / 6).astype(int)
        self.model_accuracy['binned_trial'].replace({1: 'Early', 2: 'Mid-Early', 3: 'Mid-Late', 4: 'Late'}, inplace=True)
        self.model_choice_rate['participant_id'] = np.arange(1, len(self.model_choice_rate) + 1)
            
        self.model_choice_rate = self.model_choice_rate.melt(id_vars=[self.group_code, 'participant_id'], value_vars=['A', 'B', 'E', 'F', 'N'], var_name='symbol_name', value_name='choice_rate')
        self.model_choice_rate.rename(columns={'symbol_name': 'symbol'}, inplace=True)
        self.model_choice_rate['symbol'].replace({'A': 'High Reward', 'B': 'Low Reward', 'E': 'Low Punish', 'F': 'High Punish', 'N': 'Novel'}, inplace=True)

        #Run model simulation statistics
        statistics = Statistics(rscripts_path, load_stats)
        processing = Processing()
        plotting = Plotting()

        ## Learning accuracy
        self.model_accuracy.to_csv(f'SOMA_AL/modelling/model_behaviours_{self.split_by_group}_stats_learning_data_trials.csv', index=False)
        formula = f'accuracy~1+{self.group_code}*symbol_name*binned_trial+(1|participant_id)'
        self.model_learning_accuracy_glmm = statistics.generalized_linear_model(formula, 
                                               self.model_accuracy,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/modelling/model_behaviours_{self.split_by_group}_stats_learning_data_trials.csv",
                                               savename=f"SOMA_AL/modelling/model_behaviours_{self.split_by_group_id}_stats_learning_data_trials.csv",
                                               family='binomial')
        
        comparisons = [['chronic pain', 'no pain'], ['chronic pain', 'acute pain']]
        data = processing.average_byfactor(self.model_accuracy, 'accuracy', self.group_code)
        self.model_learning_accuracy_planned_group = statistics.planned_ttests('accuracy', self.group_code, comparisons, data)

        comparisons = [['chronic pain~Reward', 'no pain~Reward'], 
                    ['chronic pain~Punish', 'no pain~Punish'], 
                    ['chronic pain~Reward-Punish', 'no pain~Reward-Punish']]
        factors = [self.group_code, 'symbol_name']
        data1 = self.average_byfactor(self.model_accuracy, 'accuracy', factors)
        data2 = self.manipulate_data(data1, 'accuracy', 'symbol_name', 'Reward-Punish')
        data = [data1, data1, data2]
        self.model_learning_accuracy_planned_interaction = self.planned_ttests('accuracy', factors, comparisons, data)

        ## Choice rate
        self.model_choice_rate.to_csv(f'SOMA_AL/modelling/model_behaviours_{self.split_by_group}_stats_transfer_data.csv', index=False)
        formula = f'choice_rate~1+{self.group_code}*symbol+(1|participant_id)'
        self.model_transfer_choice_rate_glmm = statistics.generalized_linear_model(formula, 
                                        self.model_choice_rate,
                                        path=self.repo_directory,
                                        filename=f"SOMA_AL/modelling/model_behaviours_{self.split_by_group}_stats_transfer_data.csv",
                                        savename=f"SOMA_AL/modelling/model_behaviours_{self.split_by_group_id}_stats_transfer_data.csv",
                                        family='gaussian')
        
        data = self.average_byfactor(self.model_choice_rate, 'choice_rate', self.group_code)
        comparisons = [['chronic pain', 'no pain'], ['chronic pain', 'acute pain']]
        self.model_transfer_choice_rate_planned_group = self.planned_ttests('choice_rate', self.group_code, comparisons, data)
        
        comparisons = [['High Reward', 'Low Punish'], ['Low Reward', 'Low Punish']]
        self.model_transfer_choice_rate_planned_context = statistics.planned_ttests('choice_rate', 'symbol', comparisons, self.model_choice_rate)

        comparisons = [['no pain~High Reward-Low Punish', 'acute pain~High Reward-Low Punish'],
                       ['no pain~High Reward-Low Punish', 'chronic pain~High Reward-Low Punish'],
                       ['acute pain~High Reward-Low Punish', 'chronic pain~High Reward-Low Punish'],
       
                       ['no pain~Low Reward-Low Punish', 'acute pain~Low Reward-Low Punish'],
                       ['no pain~Low Reward-Low Punish', 'chronic pain~Low Reward-Low Punish'],
                       ['acute pain~Low Reward-Low Punish', 'chronic pain~Low Reward-Low Punish']]
        
        data = self.average_byfactor(self.model_choice_rate, 'choice_rate', [self.group_code, 'symbol'])
        factors = [self.group_code, 'symbol']
        data1 = self.manipulate_data(data, 'choice_rate', 'symbol', 'High Reward-Low Punish')
        data2 = self.manipulate_data(data, 'choice_rate', 'symbol', 'Low Reward-Low Punish')
        data = [data1, data1, data1, data2, data2, data2] if self.split_by_group == 'pain' else [data1, data2]
        self.model_transfer_accuracy_planned_interaction = self.planned_ttests('choice_rate', factors, comparisons, data)

        #post-hoc comparisons

        #Model Results (parameter statistics)
        self.model_parameters_glmm_summary = pd.read_csv(f'SOMA_AL/modelling/{self.split_by_group}_fits_linear_results.csv')
        self.model_parameters_glmm_summary = self.model_parameters_glmm_summary[self.model_parameters_glmm_summary['model']==self.best_model]
        self.model_parameters_glmm_summary.rename(columns={'parameter': 'factor', 'df_model': 'df', 'F': 'test_value'}, inplace=True)

        parameter_metadata = {
            'formula': 'prameter~group_code',
            'fixed_effects': ['group_code'],
            'random_effects': '',
            'outcome': 'parameter',
            'sample_size': self.model_learning_accuracy_glmm['metadata']['sample_size'],
            'test': 'F',
            'df_residual': int(self.model_parameters_glmm_summary['df_res'].values[0])
        }
        self.model_parameters_glmm = {'metadata': parameter_metadata,
                                      'model_summary': self.model_parameters_glmm_summary}
        
        self.model_parameters_planned_group_summary = pd.read_csv(f'SOMA_AL/modelling/{self.split_by_group}_fits_ttest_results.csv')
        self.model_parameters_planned_group_summary = self.model_parameters_planned_group_summary[self.model_parameters_planned_group_summary['model']==self.best_model]
        self.model_parameters_planned_group_summary.rename(columns={'parameter': 'factor'}, inplace=True)

        comparisons = self.model_parameters_planned_group_summary['comparison'].unique()
        comparisons = [comparison.split(' vs ') for comparison in comparisons]
        parameter_metadata = {
            'metric': 'prameter',
            'factor': ['group_code'],
            'comparisons': comparisons,
            'test': 't'
        }
        self.model_parameters_planned_group = {'metadata': parameter_metadata,
                                               'model_summary': self.model_parameters_planned_group_summary}
        
        self.model_parameters_posthoc_group = pd.read_csv(f'SOMA_AL/modelling/{self.split_by_group}_fits_posthoc_results.csv')

        #Parameter Correlations
        pain_scores = self.pain_scores
        with open('SOMA_AL/modelling/fit_data_FIT.pkl', 'rb') as f:
            fit_data = pickle.load(f)
        fit_data = fit_data[self.best_model]
        fit_data.rename(columns={'participant': 'participant_id'}, inplace=True)
        parameter_names = fit_data.columns[4:]
        pain_names = pain_scores.columns[2:]
        fit_data = fit_data.merge(pain_scores[['participant_id', 'intensity', 'unpleasant', 'interference']], on='participant_id', how='left')

        #Create a corellation matrix for the model parameters and pain scores
        correlation_matrix = pd.DataFrame(index=parameter_names, columns=pain_names)
        for parameter in parameter_names:
            for pain_metric in pain_names:
                correlation_data = fit_data[[parameter, pain_metric]]
                r, p = stats.pearsonr(correlation_data[parameter], correlation_data[pain_metric])
                r = str(f'{np.round(r, 2):.2f}').replace('0.','.') 
                p = str(f'{np.round(p, 4):.4f}').replace('0.','.') if p > 0.0001 else '> .0001'
                correlation_matrix.loc[parameter, pain_metric] = f'{r} ({p})'

        #Create new correlation matrices like above, but for each pain group
        group_correlation_matrix = {group: pd.DataFrame(index=parameter_names, columns=pain_names) for group in ['no pain', 'acute pain', 'chronic pain']}
        for pain_group in ['no pain', 'acute pain', 'chronic pain']:
            group_fit_data = fit_data[fit_data['pain_group'] == pain_group]
            for parameter in parameter_names:
                for pain_metric in pain_names:
                    correlation_data = group_fit_data[[parameter, pain_metric]]
                    r, p = stats.pearsonr(correlation_data[parameter], correlation_data[pain_metric])
                    r = str(f'{np.round(r, 2):.2f}').replace('0.','.') 
                    p = str(f'{np.round(p, 4):.4f}').replace('0.','.') if p > 0.0001 else '> .0001'
                    group_correlation_matrix[pain_group].loc[parameter, pain_metric] = f'{r} ({p})'

        metadata = {'formula': 'parameter~pain_score',
                    'fixed_effects': pain_names.tolist(),
                    'random_effects': '',
                    'outcome': parameter_names.tolist(),
                    'sample_size': fit_data['participant_id'].nunique(),
                    'test': 'r',
                    'df_residual': fit_data.shape[0] - 2}
        self.model_parameters_pain = {'metadata': metadata, 'model_summary': correlation_matrix, 'group_summary': group_correlation_matrix}
        
        #Plot the correlations
        plotting.plot_model_parameters_by_pain_split(fit_data, parameter_names, pain_names)