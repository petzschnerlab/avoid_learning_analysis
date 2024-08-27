
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf

class SOMAStatistics:

    def run_statistics(self):

        #Demograhpics T-Tests
        self.stats_age = self.linear_model(f'age~{self.group_code}', self.demographics)
        self.stats_intensity = self.linear_model(f'intensity~{self.group_code}', self.pain_scores)
        self.stats_unpleasant = self.linear_model(f'unpleasant~{self.group_code}', self.pain_scores)
        self.stats_interference = self.linear_model(f'interference~{self.group_code}', self.pain_scores)
        if self.depression_scores is not None:
            self.stats_depression = self.linear_model(f'depression~{self.group_code}', self.depression_scores)

        #Linear Mixed Effects Models
        #self.learning_lmem = self.linear_model(self.learning_data)

    def linear_model(self, formula, data):
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

        #Fit model
        if random_effect:
            mdf = smf.mixedlm(formula=fixed_formula, data=data, groups=random_effect).fit()
        else:
            mdf = smf.ols(formula=fixed_formula, data=data).fit()

        #Turn into pandas dataframe
        mdf_summary = mdf.summary() #TODO: HERE
        #mdf_summary = mdf_summary.tables[0].as_html()
        #mdf_summary = pd.read_html(mdf_summary, header=0, index_col=0)[0]

        return mdf_summary
    

if __name__ == '__main__':

    #TTest Example
    x_ = (373,398,245,272,238,241,134,410,158,125,198,252,577,272,208,260)
    y_ = (411,471,320,364,311,390,163,424,228,144,246,371,680,384,279,303)

    soma_stats = SOMAStatistics()
    soma_stats.t_test(x_, y_)

    #LMEM Example
    data = sm.datasets.get_rdataset("dietox", "geepack").data
    formula = 'Weight ~ Time + (1|Pig)'
    
    soma_stats = SOMAStatistics()
    soma_stats.linear_model(data, formula)


