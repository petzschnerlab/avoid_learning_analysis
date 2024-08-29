#Load modules
library(lme4)
library(lmerTest)

#Parse parameters
args = commandArgs(trailingOnly = TRUE)
working_directory = args[[1]]
filename = args[[2]]
formula = args[[3]]

#Set working directory to where the data is
setwd(working_directory) #"C:/Users/cwill/Documents/GitHub/SOMA_avoidance_learning"

#Load csv file
data = read.csv(filename, header = TRUE) #"SOMA_AL/stats/stats_learning_data.csv"

#Fit the model
model = lmer(formula, data=data) #accuracy ~ 1 + group_code*symbol_name + (1|participant_id)

#Save results as csv
save_name = gsub('.csv', '_results.csv', filename)
write.csv(anova(model), save_name)
