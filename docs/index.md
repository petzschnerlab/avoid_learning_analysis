---
hide:
- toc
---
# Avoidance Learning Analysis

Welcome to the avoidance learning analysis repo. This repo was built for the PEAC lab to analyse behavioural data from the various avoidance learning tasks. This repo loads the data, conducts statistics on the data, plots the data, and generates a report as a PDF file, which presents the main findings of the study. 

## Project Pipeline
This repo is one part of a project pipeline, which requires the coordination of multiple repos. Projects begin with a <b>task repo</b>, which is used to collect behavioural data from participants either locally or on Prolific. The collected data must then be pushed through a <b>data extraction repo</b> to prepare CSV files for analysis. These CSV files are used in <b>the analysis repo (this repo)</b>, which creates a PDF report (`AL/reports`), ending the project pipeline. 

Optionally, you can run computational reinforcement learning models using the <b>modelling repo</b>, and the results can be added to the report here. This is a bit clunky because it requires a bit of back-and-forth between this repo and the modelling repo. Specifically, this repo must be run (with `load_models=False`, see [Parameters](HowToUse/parameters.md) in documentation) in order to create two CSV files that the modelling repo needs (`AL/data/pain_learning_processed.csv` and `AL/data/pain_transfer_processed.csv`). These files can then be manually moved into the modelling repo's data directory (`RL/data`). The modelling repo can then be used to model the data, which will result in a newly constructed directory called `modelling` (`RL/modelling`). This folder can then be manually moved to this analysis repo as `AL/modelling`. Then you can re-run this repo (with `load_models=True`) and the modelling results will be included in the PDF report. 

## Project Repos

### Task Repos
There exists several versions of the avoidance learning task. This package was built around two of these repos:

- [Version 1a](https://github.com/petzschnerlab/v1a_avoid_pain) 
- [Version 1b](https://github.com/petzschnerlab/v1b_avoid_paindepression)

There also exists other task repos that are likely compatible with this analysis code, but have never been tested:

- [Version 2](https://github.com/petzschnerlab/v2_avoid_paindepression_presample)
- [Version EEG](https://github.com/petzschnerlab/soma_avoid_eeg)

### Data Extraction Repo
There also exists some code that extracts the data collected by the task repos (which come as `.json` files) and formats it into a .csv file.

- [Data Extraction](https://github.com/petzschnerlab/avoid_learning_data_extraction)

### Analysis Repo
Next there is the analysis repo, which conducts statistics, creates plots, and build a PDF report of the main findings.

- [Analysis](https://github.com/petzschnerlab/avoid_learning_analysis)

### Computational Modelling Repo
Finally, there is a companion repo to the analysis repo, which fits computational reinforcement learning models to the data. 

- [RL Modelling](https://github.com/petzschnerlab/avoid_learning_rl_models)

## Final Report and Other Findings

The report is saved as a PDF and displays the main findings of the analyses. There actually exists a lot more analyses/plots than what is presented in the report. You'll see additional statistical results and plots in the `AL/stats` and `AL/plots` folders, respectively. You can also look through the `run_statistics` method of the `Statistics` class (`AL/helpers/statistics.py`) and the `print_plots` method of the `Plotting` class (`RL/helpers/plotting.py`) to see what else is conducted. If you do this, you will also find some statistics that are never saved to a file (such as the covariate GLMM analyses). It might be worth seeing which of these stats and plots turn out to be useful across the future projects and add them to the report.