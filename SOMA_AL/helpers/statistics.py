
import os
import warnings
import pandas as pd
import subprocess
import statsmodels.api as sm
import statsmodels.formula.api as smf

class Statistics:

    def get_pvalue(self, summary):
        if self.hide_stats:
            return 'Hidden'

        p_value = summary['model_summary']['p_value'][0].round(4)
        if p_value < 0.0001:
            p_value = '<0.0001'
        else:
            p_value = str(p_value)

        return p_value
    
    def get_planned_t(self, summary):
        if self.hide_stats:
            return 'Hidden'

        planned_t = summary['model_summary']['planned_t'][0]

        return planned_t

    def run_statistics(self):

        #Demograhpics linear models
        self.stats_age = self.linear_model(f'age~{self.group_code}', self.demographics)
        self.stats_intensity = self.linear_model(f'intensity~{self.group_code}', self.pain_scores)
        self.stats_unpleasant = self.linear_model(f'unpleasant~{self.group_code}', self.pain_scores)
        self.stats_interference = self.linear_model(f'interference~{self.group_code}', self.pain_scores)
        if self.depression_scores is not None:
            self.stats_depression = self.linear_model(f'PHQ8~{self.group_code}', self.depression_scores)

        #Demographic planned t-tests
        comparisons =  [['no pain', 'acute pain'], ['no pain', 'chronic pain']]
        self.tstats_age = self.planned_ttests('age', comparisons, self.demographics)
        self.tstats_intensity = self.planned_ttests('intensity', comparisons, self.pain_scores)
        self.tstats_unpleasant = self.planned_ttests('unpleasant', comparisons, self.pain_scores)
        self.tstats_interference = self.planned_ttests('interference', comparisons, self.pain_scores)
        if self.depression_scores is not None:
            self.tstats_depression = self.planned_ttests('PHQ8', comparisons, self.depression_scores)

        #Prepare summaries for statistical reporting
        factor_labels = ['Age', 'Pain Intensity', 'Pain Unpleasantness', 'Pain Interference', 'Depression']
        self.demo_clinical = pd.concat([self.stats_age['model_summary'],
                                        self.stats_intensity['model_summary'],
                                        self.stats_unpleasant['model_summary'],
                                        self.stats_interference['model_summary']], axis=0)
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
                                    self.tstats_interference['model_summary']], axis=0)
            if self.depression_scores is not None:
                self.demo_clinical_planned = pd.concat([self.demo_clinical_planned, self.tstats_depression['model_summary']], axis=0)
            self.demo_clinical_planned = self.demo_clinical_planned.reset_index(drop=True)
            demo_clinical_labels = pd.DataFrame({'factor': [label for label in factor_labels for _ in range(2)]})
            self.demo_clinical_planned = pd.concat([demo_clinical_labels, self.demo_clinical_planned], axis=1)

            self.demo_clinical_planned = {'metadata': self.demo_metadata, 'model_summary': self.demo_clinical_planned}
    
        #Linear Mixed Effects Models
        formula = f'accuracy~1+{self.group_code}*symbol_name+(1|participant_id)'
        if self.covariate is not None:
            formula = f'accuracy~1+{self.group_code}*symbol_name+{self.covariate}+(1|participant_id)'
        self.learning_accuracy_glmm = self.linear_model(formula, 
                                               self.learning_data,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_learning_data_trials.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_learning_data_trials.csv",
                                               family='binomial')
        
        formula = f'rt~1+{self.group_code}*symbol_name+(1|participant_id)'
        if self.covariate is not None:
            formula = f'rt~1+{self.group_code}*symbol_name+{self.covariate}+(1|participant_id)'
        self.learning_rt_glmm = self.linear_model(formula, 
                                               self.learning_data,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_learning_data_trials.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_learning_data_trials.csv",
                                               family='Gamma')
        
        formula = f'accuracy~1+{self.group_code}*context+(1|participant_id)'
        if self.covariate is not None:
            formula = f'accuracy~1+{self.group_code}*context+{self.covariate}+(1|participant_id)'
        self.transfer_accuracy_glmm = self.linear_model(formula, 
                                               self.transfer_data_reduced,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_transfer_data_trials_reduced.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_transfer_data_trials_reduced.csv",
                                               family='binomial')

        formula = f'rt~1+{self.group_code}*context+(1|participant_id)'
        if self.covariate is not None:
            formula = f'rt~1+{self.group_code}*context+{self.covariate}+(1|participant_id)'
        self.transfer_rt_glmm = self.linear_model(formula, 
                                               self.transfer_data_reduced,
                                               path=self.repo_directory,
                                               filename=f"SOMA_AL/stats/{self.split_by_group}_stats_transfer_data_trials_reduced.csv",
                                               savename=f"SOMA_AL/stats/{self.split_by_group_id}_stats_transfer_data_trials_reduced.csv",
                                               family='Gamma')
        
        comparisons = [['no pain', 'acute pain'], ['no pain', 'chronic pain']]
        self.learning_accuracy_planned = self.planned_ttests('accuracy', comparisons, self.learning_data, average=True)
        self.learning_rt_planned = self.planned_ttests('rt', comparisons, self.learning_data, average=True)
        self.transfer_accuracy_planned = self.planned_ttests('accuracy', comparisons, self.transfer_data_reduced, average=True)
        self.transfer_rt_planned = self.planned_ttests('rt', comparisons, self.transfer_data_reduced, average=True)
        
        self.insert_statistics()

    def insert_statistics(self):

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

    def linear_model(self, formula, data, path=None, filename=None, savename=None, family='gaussian'):
        """
        Linear mixed effects model

        data: pandas DataFrame
        formula: string,
            formula structure: "y ~ x1 + x2 + x3 + (1|Group)"
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
            if family == 'gaussian':
                model_summary = model_summary[['Unnamed: 0', 'NumDF', 'F value', 'Pr(>F)']]
            else: 
                model_summary = model_summary[['Unnamed: 0', 'Df', 'Chisq', 'Pr(>Chisq)']]
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

        #Collect metadata
        fixed_effects = fixed_formula.split('~')[1].split('(')[0].split('+')
        fixed_effects = [fixed_effect for fixed_effect in fixed_effects if fixed_effect != '1']
        split_effects = [fixed_effect.split('*')+[fixed_effect] for fixed_effect in fixed_effects if '*' in fixed_effect]
        split_effects = split_effects[0] if len(split_effects) > 0 else split_effects
        fixed_effects = fixed_effects + split_effects
        fixed_effects = list(dict.fromkeys(fixed_effects))
        fixed_effects.sort(key=lambda x: x.count('*'))

        test = 'F' if family == 'gaussian' else 'Chisq'
        
        metadata = {'path': path, 
                    'filename': filename, 
                    'family': family, 
                    'formula': formula,
                    'outcome': formula.split('~')[0],
                    'fixed_effects': fixed_effects,
                    'random_effects': random_effect,
                    'sample_size': data['participant_id'].nunique(),
                    'df_residual': df_residual if not random_effect else None,
                    'test': test}

        return {'metadata': metadata, 'model_summary': model_summary}
    

    def planned_ttests(self, metric, comparisons, data, average=False):
            
            """
            Planned t-tests for a referent label
            [[no pain, acute pain], [no pain, chronic pain]]
            
            """

            #Average data
            if average:
                data = data.groupby(['participant_id', self.group_code])[metric].mean().reset_index()            

            #Run the t-tests
            model_summary = pd.DataFrame()
            for comparison in comparisons:

                #Referent data
                condition1_data = data[data[self.group_code] == comparison[0]]
                condition2_data = data[data[self.group_code] == comparison[1]]

                #Run the t-tests
                ttest = sm.stats.ttest_ind(condition1_data[metric], condition2_data[metric])
                ttest = pd.DataFrame({'condition1': comparison[0], 
                                    'condition2': comparison[1], 
                                    'comparison': f'{comparison[0]} vs {comparison[1]}',
                                    't_value': ttest[0], 
                                    'p_value': ttest[1],
                                    'df': ttest[2]}, index=[0])
                model_summary = pd.concat([model_summary, ttest], axis=0)

            metadata = {'metric': metric,
                        'comparisons': comparisons,
                        'test':'t'}

            return {'metadata': metadata, 'model_summary': model_summary}
