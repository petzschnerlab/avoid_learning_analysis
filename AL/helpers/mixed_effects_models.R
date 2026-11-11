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
#    The formula for the mixed effects model. Random slopes are supported, e.g. "y~1+a*b+(1+b|participant_id)".
#    Binomial count outcomes are supported as "cbind(successes,failures)~...".
#family : str
#    The family of the model ('gaussian', 'binomial', or 'Gamma'). Gamma models use a log link.
#method : str (optional)
#    How fixed effects are tested: 'LRT' (likelihood-ratio tests via afex::mixed; default for non-gaussian models)
#    or 'Wald' (Type III Wald chi-square tests via car::Anova; always used for gaussian models).

#Returns (External)
#------------------
#File: CSV
#    A CSV file containing the results of the mixed effects model (<savename>_<outcome>_results.csv).
#File: CSV
#    A CSV file containing model diagnostics: convergence messages and singular fit (<savename>_<outcome>_diagnostics.csv).

#Load modules
library(lme4)
library(lmerTest)
library(car)
library(afex)

#Sum-to-zero contrasts for all factors, so Type III tests of main effects are interpretable
#(afex::mixed already applies these; this extends them to the gaussian models fitted with lmer)
options(contrasts = c('contr.sum', 'contr.poly'))

#Parse parameters
args = commandArgs(trailingOnly = TRUE)

working_directory = args[[1]]
filename = args[[2]]
savename = args[[3]]
formula = args[[4]]
family = args[[5]]
method = if (length(args) >= 6) args[[6]] else 'LRT'

#Set working directory to where the data is
setwd(working_directory)

#Load csv file
data = read.csv(filename, header = TRUE)
if ('accuracy' %in% colnames(data)){
  data$accuracy[data$accuracy == 100] = 1
}

#Treat character columns as factors (sum-to-zero contrasts are applied to factors)
for (column in colnames(data)) {
  if (is.character(data[[column]])) data[[column]] = factor(data[[column]])
}

#Outcome label used in the output file names (for cbind(a,b) outcomes, the first column name)
outcome = strsplit(formula, '~')[[1]][1]
outcome = sub('^cbind\\(([^,]+),.*$', '\\1', outcome)

#Model family (Gamma uses a log link)
model_family = switch(family,
                      'binomial' = binomial(link = 'logit'),
                      'Gamma' = Gamma(link = 'log'),
                      'gamma' = Gamma(link = 'log'),
                      NULL)

#Optimizer settings for the (larger) random slopes models
glmer_control = glmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 2e5))
lmer_control = lmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 2e5))

#Run the model
if (family == 'None' | family == 'gaussian'){
  model = lmer(formula, data=data, REML=FALSE, control=lmer_control)
  results = Anova(model, type = 3, test.statistic="Chisq")
} else if (method == 'Wald') {
  model = glmer(as.formula(formula), data=data, family=model_family, control=glmer_control)
  results = Anova(model, type = 3, test.statistic="Chisq")
} else {
  fit = afex::mixed(as.formula(formula), data=data, family=model_family, method='LRT', control=glmer_control, progress=FALSE)
  results = fit$anova_table
  model = fit$full_model
}

#Collect diagnostics of the full model
messages = c(model@optinfo$conv$lme4$messages, model@optinfo$warnings)
diagnostics = data.frame(formula = formula,
                         family = if (is.null(model_family)) 'gaussian' else paste(model_family$family, model_family$link),
                         method = if (family == 'None' | family == 'gaussian') 'Wald' else method,
                         n_obs = nobs(model),
                         singular = isSingular(model),
                         convergence = if (length(messages) == 0) 'converged' else paste(unlist(messages), collapse=' | '))

#Save results as csv
save_name = gsub('.csv', paste('_',outcome,'_results.csv',sep=''), savename)
write.csv(results, save_name)
write.csv(diagnostics, gsub('.csv', paste('_',outcome,'_diagnostics.csv',sep=''), savename), row.names = FALSE)
