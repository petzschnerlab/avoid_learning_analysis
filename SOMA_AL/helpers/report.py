#Import modules
import os
import warnings
import pandas as pd
import dataframe_image as dfi
from markdown_pdf import MarkdownPdf, Section

class Report:

    """
    Class to hold reporting functions for the SOMA project
    """

    #Report builders
    def build_report(self) -> None:

        """
        Builds and prints the SOMA report

        Returns (External)
        ------------------
        Report: PDF
            The SOMA report
        """

        #Initiate processes
        self.print_plots()
        self.pdf = MarkdownPdf(toc_level=3)

        #Add metadata
        self.insert_title_page()
        self.insert_report_details()
        self.insert_analysis_details()

        #Demographics
        self.insert_demographics_table()
        section_text = self.insert_table(self.demographics, 'demo-scores')
        section_text.extend(self.insert_image('demo-clinical-scores'))
        section_text.extend(self.get_statistics('demographics-and-clinical-scores'))
        self.add_data_pdf(section_text, center=True)
        
        #Results
        section_text = []
        section_text.append(f'## Results')
        section_text.append(f'### Learning Accuracy')
        section_text.extend(self.insert_image('learning-accuracy-by-group'))
        section_text.extend(self.insert_image('learning-accuracy'))

        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.extend(self.get_statistics('learning-accuracy'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Learning Reaction Time')
        section_text.extend(self.insert_image('learning-rt-by-group'))
        section_text.extend(self.insert_image('learning-rt'))

        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.extend(self.get_statistics('learning-rt'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Choice Rate')
        section_text.extend(self.insert_image('transfer-choice-rate'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.extend(self.get_statistics('transfer-choice-rate'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Transfer Reaction Time')   
        section_text.extend(self.insert_image('transfer-rt'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.extend(self.get_statistics('transfer-rt'))
        self.add_data_pdf(section_text, center=True)

        #Post-hocs
        if self.hide_posthocs == False:
            section_text = []
            section_text.append('## Appendix A')
            section_text.append("""Appendix A contains a collection of all possible post-hoc comparisons. 
                                Some of these are the same as the planned comparisons and should be ignored.
                                In addition, many of these comparisons are not meaningful, but we included them for completeness.""")
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('## Post-Hoc Comparisons: Group Comparisons')
            section_text.append('### Learning Accuracy')
            section_text.extend(self.insert_table(self.learning_accuracy_posthoc_group, 'learning_accuracy_group'))
            section_text.append('### Learning Reaction Time')
            section_text.extend(self.insert_table(self.learning_rt_posthoc_group, 'learning_rt_group'))
            section_text.append('### Transfer Accuracy')
            section_text.extend(self.insert_table(self.transfer_accuracy_posthoc_group, 'transfer_accuracy_group'))
            section_text.append('### Transfer Reaction Time')
            section_text.extend(self.insert_table(self.transfer_rt_posthoc_group, 'transfer_rt_group'))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('## Post-Hoc Comparisons: Context Comparisons')
            section_text.append('### Transfer Accuracy')
            section_text.extend(self.insert_table(self.transfer_accuracy_posthoc_context, 'transfer_accuracy_context'))
            section_text.append('### Transfer Reaction Time')
            section_text.extend(self.insert_table(self.transfer_rt_posthoc_context, 'transfer_rt_context'))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('## Post-Hoc Comparisons: Trial Comparisons')
            section_text.append('### Learning Accuracy')
            section_text.extend(self.insert_table(self.learning_accuracy_posthoc_trials, 'learning_accuracy_trial'))
            section_text.append('### Learning Reaction Time')
            section_text.extend(self.insert_table(self.learning_rt_posthoc_trials, 'learning_rt_trial'))
            self.add_data_pdf(section_text, center=True)

            max_rows = 30
            section_text = []
            section_text.append('## Post-Hoc Comparisons: Interaction Comparisons')
            section_text.append('### Learning Accuracy: Group x Context')
            section_text.extend(self.insert_table(self.learning_accuracy_posthoc_group_context, 'learning_accuracy_group_context', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Accuracy: Group x Trial')
            section_text.extend(self.insert_table(self.learning_accuracy_posthoc_group_trial, 'learning_accuracy_group_trial', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Accuracy: Context x Trial')
            section_text.extend(self.insert_table(self.learning_accuracy_posthoc_context_trial, 'learning_accuracy_context_trial', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Accuracy: Group x Context x Trial')
            section_text.extend(self.insert_table(self.learning_accuracy_posthoc_group_context_trial, 'learning_accuracy_group_context_trial', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Reaction Time: Group x Context')
            section_text.extend(self.insert_table(self.learning_rt_posthoc_group_context, 'learning_rt_group_context', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Reaction Time: Group x Trial')
            section_text.extend(self.insert_table(self.learning_rt_posthoc_group_trial, 'learning_rt_group_trial', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Reaction Time: Context x Trial')
            section_text.extend(self.insert_table(self.learning_rt_posthoc_context_trial, 'learning_rt_context_trial', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Learning Reaction Time: Group x Context x Trial')
            section_text.extend(self.insert_table(self.learning_rt_posthoc_group_context_trial, 'learning_rt_group_context_trial', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Transfer Accuracy')
            section_text.extend(self.insert_table(self.transfer_accuracy_posthoc_interaction, 'transfer_accuracy_interaction', max_rows))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Transfer Reaction Time')
            section_text.extend(self.insert_table(self.transfer_rt_posthoc_interaction, 'transfer_rt_interaction', max_rows))
            self.add_data_pdf(section_text, center=True)

        #Save to pdf
        self.save_report()

    def save_report(self) -> None:

        """
        Saves the report as a pdf

        Returns (External)
        ------------------
        Report: PDF
            The SOMA report
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
        t_type = '<sub>Welch</sub>' if model_summary['homogeneity_assumption'][0] == 'violated' else ''
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
            A section of the SOMA report
        """

        #Formatting
        user_css = 'h4 {text-align:center;}' if center else None
        section = Section(' \n '.join(content), toc=toc)
        self.pdf.add_section(section, user_css=user_css)

    def table_to_png(self, table: pd.DataFrame, save_name: str = "SOMA_AL/plots/Table.png") -> None:

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
                            'transfer-valence-bias': self.transfer_valence_bias}

        data = self.data_legend[target]
        formula, outcome, fixed, random, sample_size, df_residual, test = self.get_metadata(data)
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

                                        'demographics-and-clinical-scores': self.demo_clinical_planned}
        else:
            self.data_planned_legend = {'transfer-choice-rate-by-context': self.transfer_accuracy_planned_context,
                                        'transfer-rt-by-context': self.transfer_rt_planned_context,

                                        'learning-accuracy-by-interaction': self.learning_accuracy_planned_interaction,
                                        'learning-rt-by-interaction': self.learning_rt_planned_interaction,
                                        'transfer-choice-rate-by-interaction': self.transfer_accuracy_planned_interaction,
                                        'transfer-rt-by-interaction': self.transfer_rt_planned_interaction}
            
        subsection = f'**{target.replace("-", " ").title()} Statistics**\n\n'
        if data['metadata']['outcome'] == 'metric':
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
            if 'binned_trial' in factor or 'depression' == factor:
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
    def insert_image(self, image_name: str) -> list[str]:

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

        subsection = [f'#### ![{image_name}](SOMA_AL/plots/{image_name}.png)\n', 
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

        section_text = [f'# SOMA Report',
                        f'![SOMA_logo](SOMA_AL/media/SOMA_preview.png)']
        self.add_data_pdf(section_text)

    def insert_report_details(self) -> None:

        """
        Inserts the report details for the report

        Returns (External)
        ------------------
        Report: PDF
            Adds the report details to the SOMA report
        """

        section_text = [f'## SOMA Report Details',
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
            Adds the analysis details to the SOMA report
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
                        f'**Number of Participants Excluded (Accuracy Threshold: {self.accuracy_threshold}%):** {self.participants_excluded_accuracy}\n',
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