from markdown_pdf import MarkdownPdf
from helpers.report_functions import ReportFunctions

class Report(ReportFunctions):

    def __init__(self):
        super().__init__()

    #Report builders
    def build_report(self, rscripts_path=None, load_stats=False, load_models=True) -> None:

        """
        Builds and prints the SOMA report

        Parameters
        ----------
        rscripts_path : str, optional
            The path to the Rscript executable, by default None
        load_stats : bool, optional
            Whether to load the statistics from the backend if pre-run, by default False
        load_models : bool, optional
            Whether to load the modelling results from the backend if pre-run, by default True

        Returns (External)
        ------------------
        Report: PDF
            The SOMA report
        """

        #Initiate processes
        self.print_plots()
        if load_models:
            self.load_modelling_results(rscripts_path, load_stats)

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
        section_text.append(f'## Empirical Results: Accuracy and Choice Rates')
        section_text.extend(self.insert_image('empirical-performance'))
        self.add_data_pdf(section_text, center=True)
        
        section_text = []
        section_text.append('### Learning Accuracy Statistics')
        section_text.extend(self.get_statistics('learning-accuracy'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append('### Transfer Choice Rate Statistics')
        section_text.extend(self.get_statistics('transfer-choice-rate'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.append(f'## Empirical Results: Reaction Times')
        section_text.extend(self.insert_image('empirical-rt'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.extend(self.get_statistics('learning-rt'))
        self.add_data_pdf(section_text, center=True)

        section_text = []
        section_text.extend(self.get_statistics('transfer-rt'))
        self.add_data_pdf(section_text, center=True)

        #Modelling Results
        if load_models:
            section_text = []
            section_text.append('## Modelling Results')
            section_text.append('### Model Fits')
            section_text.extend(self.insert_table(self.model_AIC, 'model-AIC'))
            section_text.extend(self.insert_table(self.model_BIC, 'model-BIC'))
            section_text.extend(self.insert_table(self.model_AIC_percentages, 'model-AIC-percentages'))
            section_text.extend(self.insert_table(self.model_BIC_percentages, 'model-BIC-percentages'))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Number of Runs')
            filename = 'AL/modelling/fit-by-runs.png'
            section_text.extend(self.insert_image('fit-by-runs', filename))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Parameter Recovery')
            for model_name in self.model_names:
                filename = f'AL/modelling/correlations/{model_name}_correlation_plot.png'
                section_text.extend(self.insert_image(f'{model_name}-correlation-plot', filename))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Model Recovery')
            filename = f'AL/modelling/model_recovery.png'
            section_text.extend(self.insert_image('model-recovery', filename))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            best_model_name = self.best_model.split("+")[0].replace('Hybrid2','Hybrid 2').title()
            section_text.append(f'## Modelling Results: {best_model_name}')
            model_introduction = f"""The {best_model_name} model was selected as the best fitting model based on the BIC values.
            This model was able to accurately recover the parameters of the simulated data and was able to recover the model structure from the simulated data.
            The model was able to accurately predict the learning and transfer behaviours of the participants in the study.
            Thus, the following will only describe the results of the {best_model_name} model."""
            section_text.append(model_introduction)
            section_text.append(f'### Model Fit')
            filename = f'AL/modelling/model_behaviours/{self.best_model}_model_behaviours.png'
            section_text.extend(self.insert_image('model-behaviour', filename))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append(f'### Model Accuracy')
            section_text.extend(self.get_statistics('learning-model-behaviour-accuracy'))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append(f'### Model Choice Rate')
            section_text.extend(self.get_statistics('transfer-model-behaviour-choice-rate'))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.append('### Model Parameters')
            filename = f'AL/modelling/parameter_fits/{self.best_model}-model-fits.png'
            section_text.extend(self.insert_image('model-parameters', filename))
            self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.extend(self.get_statistics('model-parameters'))
            self.add_data_pdf(section_text, center=True)
        
            for group in self.group_labels:
                section_text = []
                section_text.append(f'### Model Parameter Correlations with Pain for {group.title()} Group')
                filename = f'AL/plots/model_parameter_by_pain_{group.replace(" ","_")}.png'
                section_text.extend(self.insert_image(f'parameter-correlation-{group.replace(" ","-")}', filename))
                self.add_data_pdf(section_text, center=True)

            section_text = []
            section_text.extend(self.insert_table(self.model_parameters_pain['model_summary'], 'model-parameters-correlation-table'))
            section_text.extend(self.insert_table(self.model_parameters_pain['group_summary']['no pain'], 'model-parameters-correlation-table-no'))
            section_text.extend(self.insert_table(self.model_parameters_pain['group_summary']['acute pain'], 'model-parameters-correlation-table-acute'))
            section_text.extend(self.insert_table(self.model_parameters_pain['group_summary']['chronic pain'], 'model-parameters-correlation-table-chronic'))
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

            if load_models:
                section_text = []
                section_text.append('## Post-Hoc Comparisons: Model Parameters')
                model_posthocs = self.model_parameters_posthoc_group[self.model_parameters_posthoc_group['model'] == self.best_model]
                for parameter in model_posthocs['parameter'].unique():
                    section_text.append(f'### {parameter.replace("_"," ").replace("lr","learning rate").title()}')
                    parameter_results = model_posthocs[model_posthocs['parameter'] == parameter][['factor', 'meandiff', 'p-adj', 'reject']]
                    parameter_results.set_index('factor', inplace=True)
                    section_text.extend(self.insert_table(parameter_results, f'{parameter}-model-parameter-posthoc'))
                self.add_data_pdf(section_text, center=True)
            
        #Save to pdf
        self.save_report()