import os
import warnings
import numpy as np
import pandas as pd
import subprocess
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as sp
import matplotlib.pyplot as plt

class Statistics:

    """
    Class to run statistical analyses
    """

    def __init__(self, rscripts_path=None, load_stats=False):
        self.rscripts_path = rscripts_path
        self.load_stats = load_stats

    #Main statistics function
    def run_statistics(self) -> None:

        """
        Main function to run the statistical analyses

        Returns (Internal)
        ------------------
        self.stats_age : dict
            The statistics for age
        self.stats_intensity : dict
            The statistics for pain intensity
        self.stats_unpleasant : dict
            The statistics for pain unpleasantness
        self.stats_interference : dict
            The statistics for pain interference
        self.stats_depression : dict
            The statistics for depression
        self.tstats_age : dict
            The t-tests for age
        self.tstats_intensity : dict
            The t-tests for pain intensity
        self.tstats_unpleasant : dict
            The t-tests for pain unpleasantness
        self.tstats_interference : dict
            The t-tests for pain interference
        self.tstats_depression : dict
            The t-tests for depression
        self.demo_clinical : dict
            The clinical statistics
        self.demo_metadata : dict
            The metadata for the clinical statistics
        self.demo_clinical_planned : dict
            The planned ttests for clinical

        self.learning_accuracy_glmm : dict
            The learning accuracy GLMM
        self.learning_rt_glmm : dict
            The learning RT GLMM
        self.transfer_accuracy_glmm : dict
            The transfer accuracy GLMM
        self.transfer_rt_glmm : dict
            The transfer RT GLMM

        self.learning_accuracy_planned_group : dict
            The planned ttests for learning accuracy group
        self.learning_accuracy_posthoc_group : dict
            The posthoc tests for learning accuracy group
        self.learning_rt_planned_group : dict
            The planned ttests for learning RT group
        self.learning_rt_posthoc_group : dict
            The posthoc tests for learning RT group
        self.transfer_accuracy_planned_group : dict
            The planned ttests for transfer accuracy group
        self.transfer_accuracy_posthoc_group : dict
            The posthoc tests for transfer accuracy group
        self.transfer_rt_planned_group : dict
            The planned ttests for transfer RT group
        self.transfer_rt_posthoc_group : dict
            The posthoc tests for transfer RT group

        self.transfer_accuracy_planned_context : dict
            The planned ttests for transfer accuracy context
        self.transfer_accuracy_posthoc_context : dict
            The posthoc tests for transfer accuracy context
        self.transfer_rt_planned_context : dict
            The planned ttests for transfer RT context
        self.transfer_rt_posthoc_context : dict
            The posthoc tests for transfer RT context

        self.learning_accuracy_planned_interaction : dict
            The planned ttests for learning accuracy interaction
        self.learning_accuracy_posthoc_group_context : dict
            The posthoc tests for learning accuracy group context
        self.learning_accuracy_posthoc_group_trial : dict
            The posthoc tests for learning accuracy group trial
        self.learning_accuracy_posthoc_context_trial : dict
            The posthoc tests for learning accuracy context trial
        self.learning_accuracy_posthoc_group_context_trial : dict
            The posthoc tests for learning accuracy group context trial
        self.learning_rt_planned_interaction : dict
            The planned ttests for learning RT interaction
        self.learning_rt_posthoc_group_context : dict
            The posthoc tests for learning RT group context
        self.learning_rt_posthoc_group_trial : dict
            The posthoc tests for learning RT group trial
        self.learning_rt_posthoc_context_trial : dict
            The posthoc tests for learning RT context trial
        self.learning_rt_posthoc_group_context_trial : dict
            The posthoc tests for learning RT group context trial
        self.transfer_accuracy_planned_interaction : dict
            The planned ttests for transfer accuracy interaction
        self.transfer_accuracy_posthoc_interaction : dict
            The posthoc tests for transfer accuracy interaction
        self.transfer_rt_planned_interaction : dict
            The planned ttests for transfer RT interaction
        self.transfer_rt_posthoc_interaction : dict
            The posthoc tests for transfer RT interaction
        """

        #Demograhpics linear models
        self.pain_scores['composite'] = self.pain_scores[['intensity', 'unpleasant', 'interference']].mean(axis=1)
        self.stats_age = self.generalized_linear_model(f'age~{self.group_code}', self.demographics)
        self.stats_intensity = self.generalized_linear_model(f'intensity~{self.group_code}', self.pain_scores)
        self.stats_unpleasant = self.generalized_linear_model(f'unpleasant~{self.group_code}', self.pain_scores)
        self.stats_interference = self.generalized_linear_model(f'interference~{self.group_code}', self.pain_scores)
        self.stats_composite = self.generalized_linear_model(f'composite~{self.group_code}', self.pain_scores)
        if self.depression_scores is not None:
            self.stats_depression = self.generalized_linear_model(f'PHQ8~{self.group_code}', self.depression_scores)

        #Demographic planned t-tests
        comparisons =  [['chronic pain', 'no pain'], ['chronic pain', 'acute pain']]
        self.tstats_age = self.planned_ttests('age', self.group_code, comparisons, self.demographics)
        self.tstats_intensity = self.planned_ttests('intensity', self.group_code, comparisons, self.pain_scores)
        self.tstats_unpleasant = self.planned_ttests('unpleasant', self.group_code, comparisons, self.pain_scores)
        self.tstats_interference = self.planned_ttests('interference', self.group_code, comparisons, self.pain_scores)
        self.tstats_composite = self.planned_ttests('composite', self.group_code, comparisons, self.pain_scores)

        self.tstats_age_posthoc = self.post_hoc_tests('age', self.group_code, self.demographics)
        self.tstats_intensity_posthoc = self.post_hoc_tests('intensity', self.group_code, self.pain_scores)
        self.tstats_unpleasant_posthoc = self.post_hoc_tests('unpleasant', self.group_code, self.pain_scores)
        self.tstats_interference_posthoc = self.post_hoc_tests('interference', self.group_code, self.pain_scores)
        self.tstats_composite_posthoc = self.post_hoc_tests('composite', self.group_code, self.pain_scores)
        if self.depression_scores is not None:
            self.tstats_depression = self.planned_ttests('PHQ8', self.group_code, comparisons, self.depression_scores)

        #Prepare summaries for statistical reporting
        factor_labels = ['Age', 'Pain Intensity', 'Pain Unpleasantness', 'Pain Interference', 'Composite', 'Depression']
        self.demo_clinical = pd.concat([self.stats_age['model_summary'],
                                        self.stats_intensity['model_summary'],
                                        self.stats_unpleasant['model_summary'],
                                        self.stats_interference['model_summary'],
                                        self.stats_composite['model_summary']], axis=0)
        if self.depression_scores is not None:
            self.demo_clinical = pd.concat([self.demo_clinical, self.stats_depression['model_summary']], axis=0)
        self.demo_clinical = self.demo_clinical.reset_index(drop=True)
        for i in range(self.demo_clinical.shape[0]):
            self.demo_clinical.loc[i, 'factor'] = factor_labels[i]

        self.demo_metadata = self.stats_age['metadata'].copy()
        self.demo_metadata['formula'] = self.demo_metadata['formula'].replace('age', 'metric')
        self.demo_metadata['outcome'] = 'metric'

        self.demo_clinical = {'metadata': self.demo_metadata, 'model_summary': self.demo_clinical}

        if self.split_by_group == 'pain':
            self.demo_clinical_planned = pd.concat([self.tstats_age['model_summary'],
                                    self.tstats_intensity['model_summary'],
                                    self.tstats_unpleasant['model_summary'],
                                    self.tstats_interference['model_summary'],
                                    self.tstats_composite['model_summary']], axis=0)
            self.demo_clinical_posthoc = pd.concat([self.tstats_age_posthoc,
                                    self.tstats_intensity_posthoc,
                                    self.tstats_unpleasant_posthoc,
                                    self.tstats_interference_posthoc,
                                    self.tstats_composite_posthoc], axis=0)
            if self.depression_scores is not None:
                self.demo_clinical_planned = pd.concat([self.demo_clinical_planned, self.tstats_depression['model_summary']], axis=0)
            self.demo_clinical_planned = self.demo_clinical_planned.reset_index(drop=True)
            self.demo_clinical_posthoc = self.demo_clinical_posthoc.reset_index(drop=True)
            demo_clinical_labels = pd.DataFrame({'factor': [label for label in factor_labels for _ in range(2)]})
            self.demo_clinical_planned = pd.concat([demo_clinical_labels, self.demo_clinical_planned], axis=1)
            demo_clinical_labels = pd.DataFrame({'factor': [label for label in factor_labels for _ in range(3)]})
            self.demo_clinical_posthoc = pd.concat([demo_clinical_labels, self.demo_clinical_posthoc], axis=1)

            self.demo_clinical_planned = {'metadata': self.demo_metadata, 'model_summary': self.demo_clinical_planned}
            self.demo_clinical_posthoc = {'metadata': self.demo_metadata, 'model_summary': self.demo_clinical_posthoc}
    
        #Linear Mixed Effects Models
        #Learning accuracy
        formula = f'accuracy~1+{self.group_code}*symbol_name*binned_trial+(1|participant_id)'
        if self.covariate is not None:
            formula = f'accuracy~1+{self.group_code}*symbol_name*binned_trial+{self.covariate}+(1|participant_id)'
        
        assumption_data = self.average_byfactor(self.learning_data, 'accuracy', [self.group_code, 'symbol_name', 'binned_trial'])
        assumption_data[self.group_code] = pd.Categorical(assumption_data[self.group_code], self.group_labels)
        assumption_data['symbol_name'] = pd.Categorical(assumption_data['symbol_name'], ['Reward', 'Punish'])
        assumption_data['binned_trial'] = pd.Categorical(assumption_data['binned_trial'], ['Early', 'Mid-Early', 'Mid-Late', 'Late'])
        self.learning_accuracy_glmm_assumptions = self.glmm_assumption_check(assumption_data, formula, phase='learning')

        self.learning_accuracy_glmm = self.generalized_linear_model(formula, 
                                               self.learning_data,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_learning_data_trials.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_learning_data_trials.csv",
                                               family='binomial')
        
        #Learning RT
        formula = f'rt~1+{self.group_code}*symbol_name*binned_trial+(1|participant_id)'
        if self.covariate is not None:
            formula = f'rt~1+{self.group_code}*symbol_name*binned_trial+{self.covariate}+(1|participant_id)'

        assumption_data = self.average_byfactor(self.learning_data, 'rt', [self.group_code, 'symbol_name', 'binned_trial'])
        assumption_data[self.group_code] = pd.Categorical(assumption_data[self.group_code], self.group_labels)
        assumption_data['symbol_name'] = pd.Categorical(assumption_data['symbol_name'], ['Reward', 'Punish'])
        assumption_data['binned_trial'] = pd.Categorical(assumption_data['binned_trial'], ['Early', 'Mid-Early', 'Mid-Late', 'Late'])
        self.learning_rt_glmm_assumptions = self.glmm_assumption_check(assumption_data, formula, phase='learning')

        self.learning_rt_glmm = self.generalized_linear_model(formula, 
                                               self.learning_data,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_learning_data_trials.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_learning_data_trials.csv",
                                               family='Gamma')
        
        #Transfer choice rate using averaged data
        formula = f'choice_rate~1+{self.group_code}*symbol+(1|participant_id)'
        assumption_data = self.choice_rate.reset_index()
        assumption_data[self.group_code] = pd.Categorical(assumption_data[self.group_code], self.group_labels)
        assumption_data['symbol'] = pd.Categorical(assumption_data['symbol'], ['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel'])
        assumption_data['choice_rate'] = assumption_data['choice_rate'].astype(float)
        self.transfer_accuracy_glmm_assumptions = self.glmm_assumption_check(assumption_data, formula, phase='transfer')        
        
        self.transfer_accuracy_glmm = self.generalized_linear_model(formula, 
                                               self.choice_rate.reset_index(),
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_choice_rates.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_choice_rates.csv",
                                               family='gaussian')
        
        formula = f'choice_rate~1+{self.group_code}*symbol+context_val+(1|participant_id)'
        assumption_data = self.choice_rate_context.reset_index()
        assumption_data[self.group_code] = pd.Categorical(assumption_data[self.group_code], self.group_labels)
        assumption_data['symbol'] = pd.Categorical(assumption_data['symbol'], ['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel'])
        assumption_data['context_val'] = pd.Categorical(assumption_data['context_val'], ['Reward','Punish','Neutral'])
        assumption_data['choice_rate'] = assumption_data['choice_rate'].astype(float)
        self.transfer_accuracy_glmm_assumptions_context = self.glmm_assumption_check(assumption_data, formula, phase='transfer')

        self.transfer_accuracy_glmm_context = self.generalized_linear_model(formula, 
                                               self.choice_rate_context.reset_index(),
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_choice_rates_context.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_choice_rates_context.csv",
                                               family='gaussian')

        formula = f'choice_rate~1+{self.group_code}*symbol*context_val+(1|participant_id)'
        self.transfer_accuracy_glmm_context_interaction = self.generalized_linear_model(formula, 
                                              self.choice_rate_context.reset_index(),
                                              path=self.repo_directory,
                                              filename=f"SOMA_AL/stats/{self.split_by_group}_stats_choice_rates_context.csv",
                                              savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_choice_rates_context.csv", #Note, this will override the previous file made above
                                              family='gaussian')
        
        #Transfer choice RT using averaged data
        formula = f'choice_rt~1+{self.group_code}*symbol+(1|participant_id)'
        assumption_data = self.choice_rt.reset_index()
        assumption_data[self.group_code] = pd.Categorical(assumption_data[self.group_code], self.group_labels)
        assumption_data['symbol'] = pd.Categorical(assumption_data['symbol'], ['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel'])
        assumption_data['choice_rt'] = assumption_data['choice_rt'].astype(float)
        self.transfer_rt_glmm_assumptions = self.glmm_assumption_check(assumption_data, formula, phase='transfer')        
        
        self.transfer_rt_glmm = self.generalized_linear_model(formula, 
                                               self.choice_rt.reset_index(),
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_choice_rt.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_choice_rt.csv",
                                               family='Gamma')
        
        #Transfer valence bias
        formula = f'valence_bias~1+{self.group_code}'
        if self.covariate is not None:
            formula = f'valence_bias~1+{self.group_code}+{self.covariate}'

        assumption_data = self.valence_bias.reset_index()
        assumption_data[self.group_code] = pd.Categorical(assumption_data[self.group_code], self.group_labels)
        self.transfer_valence_bias_glmm_assumptions = self.glmm_assumption_check(assumption_data, formula, phase='transfer')

        self.transfer_valence_bias_glmm = self.generalized_linear_model(formula, 
                                        self.valence_bias.reset_index(),
                                        path=self.repo_directory,
                                        filename=f"SOMA_AL/stats/{self.split_by_group}_stats_transfer_valence_bias.csv",
                                        savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_transfer_valence_bias.csv",
                                        family='Gaussian')

        #Group factor comparisons
        
        '''

        == Pain Analyses ==
        Learning Phase:
        1. Chronic Pain vs No Pain
        2. Chronic Pain vs Acute Pain

        Transfer Phase:
        1. Chronic Pain vs No Pain
        2. Chronic Pain vs Acute Pain

        == Depression Analyses ==
        Learning Phase:
           None needed
        
        Transfer Phase:
           None needed

        '''

        comparisons = [['chronic pain', 'no pain'], 
                       ['chronic pain', 'acute pain']]
        
        data = self.average_byfactor(self.learning_data, 'accuracy', self.group_code)
        self.learning_accuracy_planned_group = self.planned_ttests('accuracy', self.group_code, comparisons, data)
        self.learning_accuracy_posthoc_group = self.post_hoc_tests('accuracy', self.group_code, data)
        
        data = self.average_transform_data(self.learning_data.copy(), 'rt', self.group_code, '1/x')
        self.learning_rt_planned_group = self.planned_ttests('rt', self.group_code, comparisons, data) 
        self.learning_rt_posthoc_group = self.post_hoc_tests('rt', self.group_code, data)
        
        data = self.average_byfactor(self.transfer_data_reduced, 'accuracy', self.group_code)
        self.transfer_accuracy_planned_group = self.planned_ttests('accuracy', self.group_code, comparisons, data)
        self.transfer_accuracy_posthoc_group = self.post_hoc_tests('accuracy', self.group_code, data)
        
        data = self.average_transform_data(self.transfer_data_reduced.copy(), 'rt', self.group_code,'1/x')
        self.transfer_rt_planned_group = self.planned_ttests('rt', self.group_code, comparisons, data)
        self.transfer_rt_posthoc_group = self.post_hoc_tests('rt', self.group_code, data)
        
        #Context factor comparisons
        
        '''

        == Pain and Depression Analyses ==
        Learning Phase:
           None needed

        Transfer Phase:
        1. High Reward vs Low Punish
        2. Low Reward vs Low Punish
        '''

        comparisons = [['High Reward', 'Low Punish'], ['Low Reward', 'Low Punish']]
        self.transfer_accuracy_planned_context = self.planned_ttests('choice_rate', 'symbol', comparisons, self.choice_rate.reset_index())
        self.transfer_accuracy_posthoc_context = self.post_hoc_tests('choice_rate', 'symbol', self.choice_rate.reset_index())
        self.transfer_rt_planned_context = self.planned_ttests('choice_rt', 'symbol', comparisons, self.choice_rt.reset_index())
        self.transfer_rt_posthoc_context = self.post_hoc_tests('choice_rt', 'symbol', self.choice_rt.reset_index())

        #Trial factor comparisons
        data = self.average_byfactor(self.learning_data, 'accuracy', 'binned_trial')
        self.learning_accuracy_posthoc_trials = self.post_hoc_tests('accuracy', 'binned_trial', data)
        data = self.average_transform_data(self.learning_data.copy(), 'rt', 'binned_trial', '1/x')
        self.learning_rt_posthoc_trials = self.post_hoc_tests('rt', 'binned_trial', data)

        #Valence bias comparisons

        '''
        
        == Pain & Depression Analyses ==
        There are no planned comparisons for valence bias
        '''

        self.transfer_accuracy_posthoc_bias = self.post_hoc_tests('valence_bias', self.group_code, self.valence_bias.reset_index())
        
        #Interactions comparisons
        
        '''

        == Pain Analyses ==

        Learning Phase:
        1. Chronic Pain vs No Pain: Reward
        2. Chronic Pain vs No Pain: Loss Avoid
        3. Chronic Pain vs No Pain: Reward-Loss Avoid

        Transfer Phase:
        1. No Pain vs Chronic Pain: High Reward - Low Punish
        2. No Pain vs Acute Pain: High Reward - Low Punish
        3. Acute Pain vs Chronic Pain: High Reward - Low Punish

        4. No Pain vs Chronic Pain: Low Reward - Low Punish
        5. No Pain vs Acute Pain: Low Reward - Low Punish
        6. Acute Pain vs Chronic Pain: Low Reward - Low Punish

        == Depression Analyses ==
        Learning Phase:
        1. Health vs Depression: Reward
        2. Health vs Depression: Loss Avoid
        3. Health vs Depression: Reward-Loss Avoid

        Transfer Phase:
        1. Health vs Depression: High Reward - Low Punish
        2. Health vs Depression: Low Reward - Low Punish
        
        '''
        
        if self.split_by_group == 'pain':
            comparisons = [['chronic pain~Reward', 'no pain~Reward'], 
                        ['chronic pain~Loss Avoid', 'no pain~Loss Avoid'], 
                        ['chronic pain~Reward-Loss Avoid', 'no pain~Reward-Loss Avoid']]
        else: #Depression
            comparisons = [['healthy~Reward', 'depressed~Reward'], 
                        ['healthy~Loss Avoid', 'depressed~Loss Avoid'], 
                        ['healthy~Reward-Loss Avoid', 'depressed~Reward-Loss Avoid']]
        
        factors = [self.group_code, 'context_val_name']
        data1 = self.average_byfactor(self.learning_data, 'accuracy', factors)
        data2 = self.manipulate_data(data1, 'accuracy', 'context_val_name', 'Reward-Loss Avoid')
        data = [data1, data1, data2]
        self.learning_accuracy_planned_interaction = self.planned_ttests('accuracy', factors, comparisons, data)
        
        factors = [self.group_code, 'context_val_name'] #Interaction - group|context
        data = self.average_byfactor(self.learning_data, 'accuracy', factors)
        self.learning_accuracy_posthoc_group_context = self.post_hoc_tests('accuracy', factors, data)

        factors = [self.group_code, 'binned_trial'] #Interaction - group|trial
        data = self.average_byfactor(self.learning_data, 'accuracy', factors)
        self.learning_accuracy_posthoc_group_trial = self.post_hoc_tests('accuracy', factors, data)

        factors = ['context_val_name', 'binned_trial'] #Interaction - context|trial
        data = self.average_byfactor(self.learning_data, 'accuracy', factors)
        self.learning_accuracy_posthoc_context_trial = self.post_hoc_tests('accuracy', factors, data)
        
        factors = [self.group_code, 'context_val_name', 'binned_trial'] #Interaction - group|context|trial
        data = self.average_byfactor(self.learning_data, 'accuracy', factors)
        self.learning_accuracy_posthoc_group_context_trial = self.post_hoc_tests('accuracy', factors, data)

        factors = [self.group_code, 'context_val_name']
        data1 = self.average_byfactor(self.learning_data, 'rt', factors)
        data2 = self.manipulate_data(data1, 'rt', 'context_val_name', 'Reward-Loss Avoid')
        data = [data1, data1, data2]
        self.learning_rt_planned_interaction = self.planned_ttests('rt', factors, comparisons, data)

        factors = [self.group_code, 'context_val_name'] #Interaction - group|context
        data = self.average_byfactor(self.learning_data, 'rt', factors)
        self.learning_rt_posthoc_group_context = self.post_hoc_tests('rt', factors, data)

        factors = [self.group_code, 'binned_trial'] #Interaction - group|trial
        data = self.average_byfactor(self.learning_data, 'rt', factors)
        self.learning_rt_posthoc_group_trial = self.post_hoc_tests('rt', factors, data)

        factors = ['context_val_name', 'binned_trial'] #Interaction - context|trial
        data = self.average_byfactor(self.learning_data, 'rt', factors)
        self.learning_rt_posthoc_context_trial = self.post_hoc_tests('rt', factors, data)

        factors = [self.group_code, 'context_val_name', 'binned_trial'] #Interaction - group|context|trial
        data = self.average_byfactor(self.learning_data, 'rt', factors)
        self.learning_rt_posthoc_group_context_trial = self.post_hoc_tests('rt', factors, data)

        if self.split_by_group == 'pain':
            comparisons = [['no pain~High Reward-Low Punish', 'acute pain~High Reward-Low Punish'],
                            ['no pain~High Reward-Low Punish', 'chronic pain~High Reward-Low Punish'],
                            ['acute pain~High Reward-Low Punish', 'chronic pain~High Reward-Low Punish'],

                            ['no pain~Low Reward-Low Punish', 'acute pain~Low Reward-Low Punish'],
                            ['no pain~Low Reward-Low Punish', 'chronic pain~Low Reward-Low Punish'],
                            ['acute pain~Low Reward-Low Punish', 'chronic pain~Low Reward-Low Punish']]
        else: #Depression
            comparisons = [['healthy~High Reward-Low Punish', 'depressed~High Reward-Low Punish'],
                           ['healthy~Low Reward-Low Punish', 'depressed~Low Reward-Low Punish']]
        
        factors = [self.group_code, 'symbol']
        data1 = self.manipulate_data(self.choice_rate.reset_index(), 'choice_rate', 'symbol', 'High Reward-Low Punish')
        data2 = self.manipulate_data(self.choice_rate.reset_index(), 'choice_rate', 'symbol', 'Low Reward-Low Punish')
        data = [data1, data1, data1, data2, data2, data2] if self.split_by_group == 'pain' else [data1, data2]
        self.transfer_accuracy_planned_interaction = self.planned_ttests('choice_rate', factors, comparisons, data)
        self.transfer_accuracy_posthoc_interaction = self.post_hoc_tests('choice_rate', factors, self.choice_rate.reset_index())

        data1 = self.manipulate_data(self.choice_rt.reset_index(), 'choice_rt', 'symbol', 'High Reward-Low Punish')
        data2 = self.manipulate_data(self.choice_rt.reset_index(), 'choice_rt', 'symbol', 'Low Reward-Low Punish')
        data = [data1, data1, data1, data2, data2, data2] if self.split_by_group == 'pain' else [data1, data2]
        self.transfer_rt_planned_interaction = self.planned_ttests('choice_rt', factors, comparisons, data)
        self.transfer_rt_posthoc_interaction = self.post_hoc_tests('choice_rt', factors, self.choice_rt.reset_index())

        self.insert_statistics()

    #Helper functions
    def insert_statistics(self) -> None:

        """
        Insert the statistics into the summaries

        Returns (Internal)
        ------------------
        self.demographics_summary : pd.DataFrame
            The demographics summary
        self.pain_summary : pd.DataFrame
            The pain summary
        self.depression_summary : pd.DataFrame
            The depression summary
        """

        #Add p-values to summaries
        demographics_results = pd.DataFrame({'p-value': [' ', f'{self.get_pvalue(self.stats_age)}', ' ']}, index=self.demographics_summary.index)
        self.demographics_summary = pd.concat([self.demographics_summary, demographics_results], axis=1)

        pain_results = pd.DataFrame({'p-value':                 [f'{self.get_pvalue(self.stats_intensity)}', 
                                                                 f'{self.get_pvalue(self.stats_unpleasant)}', 
                                                                 f'{self.get_pvalue(self.stats_interference)}']}, index=self.pain_summary.index)
        self.pain_summary = pd.concat([self.pain_summary, pain_results], axis=1)

        if self.depression_scores is not None:
            depression_results = pd.DataFrame({'p-value': [f'{self.get_pvalue(self.stats_depression)}']}, index=self.depression_summary.index)
            self.depression_summary = pd.concat([self.depression_summary, depression_results], axis=1)

    def get_pvalue(self, summary: dict) -> str:

        """
        Get and format the p-value from the summary

        Parameters
        ----------
        summary : dict
            The summary dictionary

        Returns
        -------
        str
            The formatted p-value
        """

        if self.hide_stats:
            return 'Hidden'

        p_value = summary['model_summary']['p_value'][0].round(4)
        if p_value < 0.0001:
            p_value = '<0.0001'
        else:
            p_value = str(p_value)

        return p_value
    
    def get_planned_t(self, summary: dict) -> str:

        """
        Get and format the planned t-value from the summary

        Parameters
        ----------
        summary : dict
            The summary dictionary

        Returns
        -------
        str
            The formatted planned t-value
        """

        if self.hide_stats:
            return 'Hidden'

        planned_t = summary['model_summary']['planned_t'][0]

        return planned_t

    #Statistical metrics and tests
    def contrast_code(self, data: pd.DataFrame) -> np.array:

        """
        Contrast code, changing variable to numbers that are orthogonal to each other

        Parameters
        ----------
        data : pd.DataFrame
            The data to contrast code

        Returns
        -------
        np.array
            The contrast coded data
        """

        #Contrast code categorical variables
        num_categories = len(data.cat.categories)
        contrast_codes = np.arange(1, num_categories+1)
        if len(contrast_codes) % 2 == 0:
            contrast_codes = contrast_codes + (contrast_codes > np.median(contrast_codes)).astype(int)
        contrast_codes = contrast_codes - np.median(contrast_codes)

        return contrast_codes
    
    def glmm_assumption_check(self, data: pd.DataFrame, formula: str, phase: str = 'learning') -> pd.DataFrame:

        """
        Check the assumptions of the generalized linear mixed effects model

        Parameters
        ----------
        data : pd.DataFrame
            The data to check the assumptions on
        formula : str
            The formula to check the assumptions on
        phase : str
            The phase of the experiment

        Returns
        -------
        pd.DataFrame
            The assumptions results
        """

        #Extract effects
        dependent_variable = formula.split('~')[0]
        fixed_effects = formula.split('~1+')[1].split('+')
        fixed_effects_expanded = []
        for fixed_effect in fixed_effects:
            if '(1|' in fixed_effect:
                pass
            elif '*' in fixed_effect:
                fixed_effects_expanded.extend(fixed_effect.split('*'))
            else:
                fixed_effects_expanded.append(fixed_effect)

        #Check assumptions
        assumptions_results = pd.DataFrame(columns=['factor', 'linearity', 'homoskedasticity', 'normality', 'note'])
        for effect_index, fixed_effect in enumerate(fixed_effects_expanded):

            #Compute residuals
            linear_coeffs = self.linear_model_categorical(f'{dependent_variable}~{fixed_effect}', data)
            fitted = data[fixed_effect].apply(lambda x: linear_coeffs[x]).astype(float)
            residuals = data[dependent_variable] - fitted

            #plot residuals
            plt.scatter(fitted, residuals, alpha=0.05)
            plt.xlabel('Fitted')
            plt.ylabel('Residuals')
            plt.title(f'{dependent_variable}~{fixed_effect}'.title())
            plt.axhline(y=0, color='black', linestyle='--')
            plt.savefig(f"{self.repo_directory}/SOMA_AL/stats/assumptions/{self.split_by_group_id}_assumption_{phase}_{dependent_variable.replace('_','')}_{fixed_effect.replace('_','')}_residuals.png")
            plt.savefig(f"{self.repo_directory}/SOMA_AL/stats/assumptions/{self.split_by_group_id}_assumption_{phase}_{dependent_variable.replace('_','')}_{fixed_effect.replace('_','')}_residuals.svg")
            plt.close()

            #Run linearity test
            #We will run a simple linear test vs the null hypothesis that the slope is 0
            #However, this does not cover curvilinear relationships, so it is important to also check the residuals plot
            linearity_results = self.linear_model_continous(f'fitted~residuals', 
                                                               pd.DataFrame({'fitted': fitted, 'residuals': residuals}))
            linearity_assumption = 'met' if linearity_results.pvalues['residuals'] > 0.05 else 'violated'

            #Run homoskedasticity test
            #We will run a simple Levene test to test the null hypothesis that the variances are equal
            #This is a bit of a simplification, as we should also check the residuals plot
            homoskedasticity_results = sp.stats.levene(data[dependent_variable], fitted).pvalue
            homoskedasticity_assumption = 'met' if homoskedasticity_results > 0.05 else 'violated'

            #Run normality of residuals test
            normality_results = sp.stats.shapiro(residuals)
            normality_assumption = 'met' if normality_results.pvalue > 0.05 else 'violated'

            #Add to results dataframe
            effect_results = {'factor': fixed_effect,
                              'linearity': linearity_assumption,
                              'homoskedasticity': homoskedasticity_assumption,
                              'normality': normality_assumption,
                              'note':'Check residual plots for true assessment'}
            
            assumptions_results = pd.concat([assumptions_results, pd.DataFrame(effect_results, index=[effect_index])], axis=0)
            
            
        return assumptions_results
    
    def ttest_assumption_check(self, group1: pd.Series, group2: pd.Series, test_type: str = 'independent') -> dict:

        """
        Check the assumptions of the t-test

        Parameters
        ----------
        group1 : pd.Series
            The first group to check the assumptions on
        group2 : pd.Series
            The second group to check the assumptions on
        test_type : str
            The type of t-test, either 'independent' or 'paired'

        Returns
        -------
        dict
            The assumption results
        """

        #Test for homogeneity of variance
        levene_results = sp.stats.levene(group1, group2)
        homogeneity_assumption = 'violated' if levene_results.pvalue < 0.05 else 'met'
        homogeneity_assumption = 'met' if test_type == 'paired' else homogeneity_assumption #Paired t-tests do not require homogeneity of variance
        assumption_results = {'homogeneity_assumption': homogeneity_assumption}

        #Test for normality
        if test_type == 'independent':
            normality_results_1, normality_results_2 = sp.stats.shapiro(group1), sp.stats.shapiro(group2)
            normality_met = normality_results_1.pvalue > 0.05 and normality_results_2.pvalue > 0.05
            assumption_results['normality_assumption'] = 'met' if normality_met else 'violated'
        else:
            normality_results = sp.stats.shapiro(group1-group2)
            normality_assumption = 'violated' if normality_results.pvalue < 0.05 else 'met'
            assumption_results['normality_assumption'] = normality_assumption

        return assumption_results

    def cohens_d(self, group1: pd.Series, group2: pd.Series, test_type: str = 'independent') -> float:

        """
        Calculate Cohen's d

        Parameters
        ----------
        group1 : pd.Series
            The first group to calculate Cohen's d on
        group2 : pd.Series
            The second group to calculate Cohen's d on
        test_type : str
            The type of t-test, either 'independent' or 'paired'

        Returns
        -------
        float
            Cohen's d
        """
        
        #Calculating statistics
        n1, n2 = len(group1), len(group2)

        mean1, mean2 = np.mean(group1), np.mean(group2)
        mean_diff = mean1 - mean2

        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        std_diff = np.std(group1 - group2, ddof=1)
        s1, s2 = (std1 ** 2) * (n1 - 1), (std2 ** 2) * (n2 - 1)
        npooled = n1 + n2 - 2
        pooled_std = np.sqrt((s1 + s2) / npooled)
        
        #Calculating Cohen's d
        if test_type == 'independent':
            d = mean_diff / pooled_std
        elif test_type == 'paired':
            d = mean_diff / std_diff
        else:
            raise ValueError('test_type must be either independent or paired')
    
        return d

    #Statistical models
    def linear_model_categorical(self, formula: str, data: pd.DataFrame) -> dict:
        
        """
        Fit a linear model to categorical data and return the coefficients

        Parameters
        ----------
        formula : str
            The formula to fit the model on
        data : pd.DataFrame
            The data to fit the model on

        Returns
        -------
        dict
            The coefficients of the model
        """        

        categories = data[formula.split('~')[-1]].cat.categories
        coeffs = smf.ols(formula=formula, data=data).fit().params
        coeffs_dict = {categories[0]: coeffs[0]}
        for category, coeff in zip(categories[1:], coeffs[1:]):
            coeffs_dict[category] = coeffs[0] + coeff

        return coeffs_dict
    
    def linear_model_continous(self, formula: str, data: pd.DataFrame) -> statsmodels.regression.linear_model.RegressionResultsWrapper:
        
        """
        Fit a linear model to continuous data

        Parameters
        ----------
        formula : str
            The formula to fit the model on
        data : pd.DataFrame
            The data to fit the model on

        Returns
        -------
        smf.ols
            The fitted model
        """        

        return smf.ols(formula=formula, data=data).fit()

    def generalized_linear_model(self, formula: str, data: pd.DataFrame, path: str = None, filename: str = None, savename: str = None, family: str = 'gaussian') -> dict:
        """
        Linear mixed effects model

        data: pandas DataFrame
        formula: string,
            formula structure: "y ~ x1 + x2 + x3 + (1|Group)"
        path: string,
            path to ...
        filename: string,
            filename to load statistics if load_stats=True
        savename: string,
            name of the file to save the results
        family: string,
            family of the model

        Returns
        -------
        dict
            Metadata and the model summary
        """
        
        #Format formula
        formula = formula.replace(' ', '')

        #Find random effect variable
        random_effect = [random_effect.split(')')[0].replace('1|','') for random_effect in formula.split('(')[1:]]
        if random_effect:
            random_effect = random_effect[0] #This only allows for a single random effect, which should be sufficient for our purposes

        #Remove random effect variable from formula
        fixed_formula = formula.split('+(')[0]

        #Fit the model
        if len(random_effect) > 0 and self.rscripts_path is not None:
            #This section runs an R script to fit the generalized linear mixed effects models. 
            #This is tricky because you need to have R and a few packages (lme4, lmerTest, car, afex, and emmeans) installed.
            #You must also have the path to the Rscript executable set in the rscripts_path variable, which can be a bit annoying.
            #The reason this is done in R is because the statsmodels package in Python does not provide factor level p-values for (generalized) linear mixed effects models.
            #This is worth looking into further, as there might be a parameter I have overlooked, or else there could be a different package that fits our needs.
            outcome = formula.split('~')[0]
            file_exists = os.path.isfile(filename.replace('.csv', f'_{outcome}_results.csv'))
            if not self.load_stats or not file_exists:
                if self.load_stats and not file_exists:
                    warnings.warn(f'''You requested to load rather than run the statistics using the load_stats=True parameter.
                                  However, the file {filename.replace(".csv", f"_{outcome}_results.csv")} does not exist. 
                                  So, we will run the linear mixed effects model now.''', stacklevel=2)
                _ = subprocess.call([self.rscripts_path,
                                    'SOMA_AL/helpers/mixed_effects_models.R', 
                                    path, 
                                    filename,
                                    savename, 
                                    formula,
                                    family])
            model_summary = pd.read_csv(savename.replace('.csv', f'_{outcome}_results.csv'))
            df_residual = None
            if False: #family == 'gaussian': #TODO: I am overhauling guassian..
                model_summary = model_summary[['Unnamed: 0', 'NumDF', 'F value', 'Pr(>F)']]
            else: 
                #check if it contains Chi Df
                df_label = 'Chi Df' if 'Chi Df' in model_summary.columns else 'Df'
                model_summary = model_summary[['Unnamed: 0', df_label, 'Chisq', 'Pr(>Chisq)']]
            model_summary.columns = ['factor', 'df', 'test_value', 'p_value']
        else:
            if random_effect:
                #Add warning
                warnings.warn(f'''This analysis {formula} contains a random effect {random_effect}, however, the rscripts_path is not set. 
                            Running linear mixed effects models requires R with the lme4, lmerTest, car, afex, and emmeans packages installed.
                            You must then provide the path to the Rscript executable in the rscripts_path variable.
                            Here is an example of where the rscripts executable may live: C:/Program Files/R/R-4.4.1/bin/x64/Rscript.
                            Until these steps are fulfilled, this analysis will proceed as a multiple regression *without* the random effect:
                            {fixed_formula}\n''', stacklevel=2)
            model_results = smf.ols(formula=fixed_formula, data=data).fit()
            model_summary = sm.stats.anova_lm(model_results, type=3)
            df_residual = model_results.df_resid
            model_summary = model_summary.reset_index()[['index', 'df', 'F', 'PR(>F)']][:-1]
            model_summary.columns = ['factor', 'df', 'test_value', 'p_value']

        #Remove the intercept
        model_summary = model_summary[model_summary['factor'] != '(Intercept)']

        #Collect metadata
        fixed_effects = fixed_formula.split('~')[1].split('(')[0].split('+')
        fixed_effects = [fixed_effect for fixed_effect in fixed_effects if fixed_effect != '1']
        split_effects = [fixed_effect.split('*')+[fixed_effect] for fixed_effect in fixed_effects if '*' in fixed_effect]
        split_effects = split_effects[0] if len(split_effects) > 0 else split_effects
        fixed_effects = fixed_effects + split_effects
        fixed_effects = list(dict.fromkeys(fixed_effects))
        fixed_effects.sort(key=lambda x: x.count('*'))

        test = 'F' if len(random_effect) ==0 else 'Chisq'
        
        metadata = {'path': path, 
                    'filename': filename, 
                    'family': family, 
                    'formula': formula,
                    'outcome': formula.split('~')[0],
                    'fixed_effects': fixed_effects,
                    'random_effects': random_effect,
                    'sample_size': data['participant_id'].nunique(),
                    'df_residual': df_residual,
                    'test': test}

        return {'metadata': metadata, 'model_summary': model_summary}

    def planned_ttests(self, metric: str, factor: str, comparisons: list[list[str]], data: pd.DataFrame) -> dict:
            
            """
            Perform planned t-tests on the data

            Parameters
            ----------
            metric : str
                The metric to perform the t-tests on
            factor : str
                The factor to perform the t-tests on
            comparisons : list[list[str]]
                The comparisons to perform the t-tests on
            data : pd.DataFrame
                The data to perform the t-tests on

            Returns
            -------
            dict
                The metadata and the model summary
            """          

            #Wrap data into a list of dataframes
            data = [data] if type(data) is not list else data
            data = data*len(comparisons) if len(data) < len(comparisons) else data

            #Run the t-tests
            model_summary = pd.DataFrame()
            for dataframe, comparison in zip(data, comparisons):

                #Get data
                if '~' in comparison[0]:
                    condition1_index = (dataframe[factor[0]] == comparison[0].split('~')[0]) & (dataframe[factor[1]] == comparison[0].split('~')[1])
                    condition2_index = (dataframe[factor[0]] == comparison[1].split('~')[0]) & (dataframe[factor[1]] == comparison[1].split('~')[1])
                    condition1_data = dataframe[condition1_index]
                    condition2_data = dataframe[condition2_index]
                else:
                    condition1_data = dataframe[dataframe[factor] == comparison[0]]
                    condition2_data = dataframe[dataframe[factor] == comparison[1]]

                #Are there the same participant ids in condition1_data and condition2_data?
                if len(set(condition1_data['participant_id']).intersection(set(condition2_data['participant_id']))) == condition1_data['participant_id'].shape[0]:
                    condition1_data = condition1_data.sort_values('participant_id')[metric].reset_index(drop=True)
                    condition2_data = condition2_data.sort_values('participant_id')[metric].reset_index(drop=True)
                    
                    assumption_check = self.ttest_assumption_check(condition1_data, condition2_data, test_type='paired')
                    ttest = sp.stats.ttest_rel(condition1_data, condition2_data)
                    cohens_d = self.cohens_d(condition1_data, condition2_data, test_type='paired')

                else:
                    condition1_data = condition1_data[metric].astype(float)
                    condition2_data = condition2_data[metric].astype(float)

                    assumption_check = self.ttest_assumption_check(condition1_data, condition2_data, test_type='independent')
                    equal_var = assumption_check['homogeneity_assumption'] == 'met'
                    ttest = sp.stats.ttest_ind(condition1_data, condition2_data, equal_var=equal_var)
                    cohens_d = self.cohens_d(condition1_data, condition2_data, test_type='independent')

                ttest = pd.DataFrame({'condition1': comparison[0], 
                                    'condition2': comparison[1], 
                                    'comparison': f'{comparison[0]} vs {comparison[1]}',
                                    't_value': ttest.statistic, 
                                    'p_value': ttest.pvalue,
                                    'cohens_d': cohens_d,
                                    'df': ttest.df,
                                    'homogeneity_assumption': assumption_check['homogeneity_assumption'],
                                    'normality_assumption': assumption_check['normality_assumption']},
                                    index=[0])
                model_summary = pd.concat([model_summary, ttest], axis=0)

            metadata = {'metric': metric,
                        'factor': factor,
                        'comparisons': comparisons,
                        'test':'t'}

            return {'metadata': metadata, 'model_summary': model_summary}

    def post_hoc_tests(self, metric: str, factor: str, data: pd.DataFrame) -> pd.DataFrame:

        """
        Perform post-hoc tests on the data

        Parameters
        ----------
        metric : str
            The metric to perform the post-hoc tests on
        factor : str
            The factor to perform the post-hoc tests on
        data : pd.DataFrame
            The data to perform the post-hoc tests on

        Returns
        -------
        pd.DataFrame
            The post-hoc test results
        """
        
        #Create combined factor
        if type(factor) is list:
            data['factor'] = data.apply(lambda x: ' & '.join([str(x[f]) for f in factor]), axis=1)
            factor = 'factor'

        #Remove any nans
        if data[metric].astype(float).isnull().sum() > 0:
            data = data.dropna(subset=[metric])

        #Run the post-hoc tests
        tukey = sm.stats.multicomp.pairwise_tukeyhsd(data[metric].astype(float), data[factor].astype("string"))._results_table.data
        tukey_table = pd.DataFrame(tukey[1:], columns=tukey[0])
        tukey_table['factor'] = tukey_table.apply(lambda x: str(x['group1']) + ' vs ' + str(x['group2']), axis=1)
        tukey_table = tukey_table.drop(columns=['group1', 'group2'])
        tukey_table.set_index('factor', inplace=True)
        tukey_table = tukey_table.drop(columns=['lower', 'upper'])

        #Run Cohen's d for each comparison
        tukey_table['cohens_d'] = np.nan
        for comparison in tukey_table.index:
            condition1_data = data[data[factor] == comparison.split(' vs ')[0]]
            condition2_data = data[data[factor] == comparison.split(' vs ')[1]]

            if len(set(condition1_data['participant_id']).intersection(set(condition2_data['participant_id']))) == condition1_data['participant_id'].shape[0]:
                condition1_data = condition1_data.sort_values('participant_id')[metric].astype(float).reset_index(drop=True)
                condition2_data = condition2_data.sort_values('participant_id')[metric].astype(float).reset_index(drop=True)
                cohens_d = self.cohens_d(condition1_data, condition2_data, test_type='paired')
            else:
                condition1_data = condition1_data[metric].astype(float)
                condition2_data = condition2_data[metric].astype(float)
                cohens_d = self.cohens_d(condition1_data, condition2_data, test_type='independent')
            
            tukey_table.at[comparison, 'cohens_d'] = cohens_d

        return tukey_table