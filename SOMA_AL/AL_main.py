import sys
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

from helpers.pipeline import Pipeline

if __name__ == '__main__':

    """
    Run the PEAC Lab AL pipeline
    """

    #Create a dict of args
    kwargs = {'author':                         'Chad C. Williams',
              'rscripts_path':                  'C:/Program Files/R/R-4.4.1/bin/x64/Rscript',
              'file_path':                      r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific',
              'file_name':                      [r'v1a_avoid_pain\v1a_avoid_pain.csv', r'v1b_avoid_paindepression\v1b_avoid_paindepression.csv'],

              'rolling_mean':                   5,
              'accuracy_exclusion_threshold':   70,
              'RT_low_threshold':               200,
              'RT_high_threshold':              5000,

              'hide_stats':                     False,
              'hide_posthocs':                  False,
              'load_stats':                     True,
              'load_posthocs':                  True,
              'load_models':                    True,
    }

    #Run the pipeline
    pipeline = Pipeline()
    pipeline.run(**kwargs)