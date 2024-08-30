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

    def add_data_pdf(self, content:list, toc:bool=True, center:bool=False):
        #Formatting
        user_css = 'h4 {text-align:center;}' if center else None
        section = Section(' \n '.join(content), toc=toc)
        self.pdf.add_section(section, user_css=user_css)

    def table_to_png(self, table:pd.DataFrame, save_name="SOMA_AL/plots/Table.png"):
        
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
        dfi.export(table, save_name, table_conversion="selenium", max_rows=-1)#, table_conversion='matplotlib')
    
    def add_figure_caption(self, text):
        section_text = f'**Figure {self.figure_count}.** {text}'
        self.figure_count += 1

        return section_text
    
    def add_table_caption(self, text):
        section_text = f'**Table {self.table_count}.** {text}'
        self.table_count += 1
        
        return section_text

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
    

    def get_caption(self, target, target_type='figure'):
        match target:
        
            case 'demo-scores':
                target_type = 'table'
                caption = 'Demographic information for each group.'

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
                Choice rate is computed as the number of times a symbol was chosen given the number of times it was presented.
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

        #Return caption
        if target_type == 'figure':
            caption = self.add_figure_caption(caption)
        else:
            caption = self.add_table_caption(caption)

        return caption
    
    def insert_image(self, image_name):
        subsection = [f'#### ![{image_name}](SOMA_AL/plots/{image_name}.png)\n', 
                      f'{self.get_caption(image_name)}\n']
       
        return subsection
    
    def insert_table(self, table_name):
        self.table_to_png(self.demographics, save_name=f'SOMA_AL/plots/{table_name}.png')

        subsection = [f'{self.get_caption(table_name)}',
                      f'#### ![{table_name}](SOMA_AL/plots/{table_name}.png)\n']
       
        return subsection
    
    def get_metadata(self, data):
        formula = data['metadata']['formula']
        fixed_effects = data['metadata']['fixed_effects']
        random_effects = data['metadata']['random_effects']
        outcome = data['metadata']['outcome']

        return formula, outcome, fixed_effects, random_effects
    
    def get_statistics(self, target):

        self.data_legend = {'learning-accuracy': self.learning_accuracy,
                            'learning-rt': self.learning_rt,
                            'transfer-choice-rate': self.transfer_accuracy,
                            'transfer-rt': self.transfer_rt}

        data = self.data_legend[target]
        formula, outcome, fixed, random = self.get_metadata(data)
        phase = target.split('-')[0]
        summary = data['model_summary']

        subsection = f'**{target.replace("-", " ").title()} Statistics**\n\n'
        subsection += f"""{outcome.capitalize()} in the {phase} phase was modelled using a linear mixed effects model with the following formula: {formula.replace('*', ':')}, 
        where {', '.join([f.replace('*',':') for f in fixed])} are the fixed effects {f'and {random} is the random effect.' if random else '.'}"""

        #Iterate through summary and format each row into a sentence
        for i, factor in enumerate(summary['factor'].unique()):
            factor_data = summary[summary['factor']==factor]
            significance = 'significant,' if (factor_data['p_value'].values < 0.05) else 'not significant,'
            df_1 = factor_data['df'].values[0]
            df_2 = '__' #TODO: populate this
            test_value = factor_data['test_value'].values.round(2)[0]
            p = factor_data['p_value'].values.round(3)[0] if (factor_data['p_value'] >= 0.001).values else '<0.001'

            if self.hide_stats:
                significance = 'hidden'
                df_1 = '__'
                df_2 = '__'
                test_value = 'hidden'
                p = 'hidden'

            subsection += f"\n\n**{factor.replace('*',':')}:** {significance}, F({df_1}, {df_2}) = {test_value}, p = {p}"
            if i == len(summary['factor'].unique())-1:
                subsection += '.'

        return [subsection]
    
    def insert_title_page(self):
        section_text = [f'# SOMA Report',
                        f'![SOMA_logo](SOMA_AL/media/SOMA_preview.png)']
        self.add_data_pdf(section_text)

    def insert_report_details(self):
        section_text = [f'## SOMA Report Details',
                        f'**Generated by:** {self.author}\n',
                        f'**Date:** {str(pd.Timestamp.now()).split(" ")[0]}',
                        #Add a description of all of the kwargs
                        f'\n### Inputted Parameters']
        for key, value in self.kwargs.items():
            section_text += [f'**{key.replace("_", " ").title()}:** {value}\n']
        section_text.append(f'\n\n{"*Note: statistics are hidden, to reveal them, set hide_stats=False.*" if self.hide_stats else ""}')
        self.add_data_pdf(section_text)

    def insert_analysis_details(self):
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
    
    def insert_demographics_table(self):
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
    
    def build_report(self):

        #Initiate processes
        self.print_plots()
        self.pdf = MarkdownPdf(toc_level=3)

        #Add metadata
        self.insert_title_page()
        self.insert_report_details()
        self.insert_analysis_details()

        #Demographics
        self.insert_demographics_table()
        section_text = self.insert_table('demo-scores')
        section_text.extend(self.insert_image('demo-clinical-scores'))
        self.add_data_pdf(section_text, center=True)
        
        #Results
        section_text = []
        section_text.append(f'## Results')
        section_text.append(f'### Learning Accuracy')
        section_text.extend(self.insert_image('learning-accuracy-by-group'))
        section_text.extend(self.insert_image('learning-accuracy-by-context'))
        section_text.extend(self.get_statistics('learning-accuracy'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Learning Reaction Time')
        section_text.extend(self.insert_image('learning-rt-by-group'))
        section_text.extend(self.insert_image('learning-rt-by-context'))
        section_text.extend(self.get_statistics('learning-rt'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Choice Rate')
        section_text.extend(self.insert_image('transfer-choice-rate'))
        section_text.extend(self.insert_image('transfer-choice-rate-neutral'))
        section_text.extend(self.get_statistics('transfer-choice-rate'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Transfer Reaction Time')   
        section_text.extend(self.insert_image('transfer-rt'))
        section_text.extend(self.insert_image('transfer-rt-neutral'))
        section_text.extend(self.get_statistics('transfer-rt'))
        self.add_data_pdf(section_text, center=True)

        #Save to pdf
        self.save_report()