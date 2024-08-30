#Load modules
library(lme4)
library(lmerTest)
library(car)
library(afex)

#Parse parameters
args = commandArgs(trailingOnly = TRUE)

working_directory = args[[1]] 
filename = args[[2]]
formula = args[[3]]
family = args[[4]]

#working_directory = 'C:/Users/cwill/Documents/GitHub/SOMA_avoidance_learning'
#filename = 'SOMA_AL/stats/stats_learning_data_trials.csv'
#formula = 'accuracy ~ 1 + group_code*symbol_name + (1|participant_id)'
#family = 'binomial'

#Set working directory to where the data is
setwd(working_directory) 

#Load csv file
data = read.csv(filename, header = TRUE)
data$accuracy[data$accuracy == 100] = 1

#Run the model
if (family == 'None' | family == 'gaussian'){
  model = lmer(formula, data=data, REML=FALSE)
  results = anova(model)
} else {
  results = afex::mixed(formula, data=data, family=family, method='LRT')$anova_table
}

#Save results as csv
outcome = strsplit(formula, '~')[[1]][1]
save_name = gsub('.csv', paste('_',outcome,'_results.csv',sep=''), filename)
write.csv(results, save_name)
