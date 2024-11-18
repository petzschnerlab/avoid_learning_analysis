#Function to run a mixed effects model on a dataset via R and save the results as a csv file.

#Parameters
#----------
#working_directory : str
#    The directory where the data is located.
#filename : str
#    The name of the csv file containing the data.
#savename : str
#    The name of the csv file to save the results to.
#formula : str
#    The formula for the mixed effects model.
#family : str
#    The family of the model (e.g., 'gaussian', 'binomial', 'poisson').

#Returns (External)
#------------------
#File: CSV
#    A CSV file containing the results of the mixed effects model.

#Load modules
library(lme4)
library(lmerTest)
library(car)
library(afex)

#Parse parameters
args = commandArgs(trailingOnly = TRUE)

working_directory = args[[1]] 
filename = args[[2]]
savename = args[[3]]
formula = args[[4]]
family = args[[5]]

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
save_name = gsub('.csv', paste('_',outcome,'_results.csv',sep=''), savename)
write.csv(results, save_name)
