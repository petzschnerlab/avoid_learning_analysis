#Load modules
from helpers.SOMA_data_processing import SOMAALPipeline

#Initiate pipeline
SOMA_pipeline = SOMAALPipeline()

#Load data
SOMA_pipeline.load_data(file_path=r'D:\BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific\v1a_avoid_pain', 
                  file_name='v1a_avoid_pain.csv')

#Summarize data
SOMA_pipeline.print_report()

#Debug tag
print()