import os
import sys
import pickle
import argparse
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from helpers.plotting import Plotting
from helpers.report_functions import ReportFunctions

"""
Lite script: re-creates ONLY the best-model parameter x pain severity scatterplots
(AL/plots/model_parameter_by_pain_{no_pain,acute_pain,chronic_pain}.png/.svg),
with learning rates log-transformed, without running the full AL pipeline.

Run from the repository root (same as AL_main.py):
    python AL/lite_parameter_pain_plots.py

Inputs (must already exist from a previous full run / the modelling package):
    AL/modelling/fit_data_FIT.pkl   - fitted parameters per model
    AL/modelling/group_BIC.csv      - used to pick the best model (same rule as the pipeline)
    The raw data files listed in DATA_FILE_NAMES - used for the pain scores
"""

#Same data settings as AL_main.py
DATA_FILE_PATH = r'AL/data'
DATA_FILE_NAMES = [r'v1a_avoid_pain.csv', r'v1b_avoid_paindepression.csv']


def get_best_model() -> str:

    """
    Select the best model exactly as the pipeline does: the lowest BIC in the last row ('full') of group_BIC.csv
    """

    model_BIC = pd.read_csv('AL/modelling/group_BIC.csv', index_col=0)
    model_BIC = model_BIC.iloc[:, :-1].round().astype(int) #drop the stored best_model column
    return model_BIC.idxmin(axis=1).values[-1]


def load_pain_scores(file_path: str, file_names: list) -> pd.DataFrame:

    """
    Load one pain score per participant, rescaled to 0-10 as in Processing.load_data, plus the composite score
    """

    data = pd.concat([pd.read_csv(os.path.join(file_path, file_name)) for file_name in file_names], axis=0)
    pain_scores = data.groupby('participant_id')[['intensity', 'unpleasant', 'interference']].first().reset_index()
    pain_scores[['intensity', 'unpleasant', 'interference']] = pain_scores[['intensity', 'unpleasant', 'interference']] / 10
    pain_scores['composite'] = pain_scores[['intensity', 'unpleasant', 'interference']].mean(axis=1)
    return pain_scores


def main(file_path: str, file_names: list, model: str = None) -> None:

    best_model = model if model is not None else get_best_model()
    print(f'Model: {best_model}')

    #Load the fitted parameters for the model
    with open('AL/modelling/fit_data_FIT.pkl', 'rb') as f:
        fit_data = pickle.load(f)[best_model]
    fit_data = fit_data.rename(columns={'participant': 'participant_id'})
    parameter_names = fit_data.columns[4:]

    #Merge in pain scores
    pain_scores = load_pain_scores(file_path, file_names)
    pain_names = pd.Index(['intensity', 'unpleasant', 'interference', 'composite'])
    fit_data = fit_data.merge(pain_scores, on='participant_id', how='left')

    n_missing = fit_data['composite'].isna().sum()
    if n_missing > 0:
        warnings.warn(f'{n_missing} fitted participant(s) have no pain scores in the data files.')

    #Log-transform learning rates and plot (same functions the pipeline uses)
    report_functions = ReportFunctions()
    log_parameters = report_functions.get_learning_rate_parameters(parameter_names)
    print(f'Log-transformed: {log_parameters}')
    plot_data = report_functions.log_transform_parameters(fit_data, log_parameters)

    os.makedirs('AL/plots', exist_ok=True)
    plotting = Plotting()
    plotting.plot_model_parameters_by_pain_split(plot_data, parameter_names, pain_names, log_parameters=log_parameters)

    for group in ['no_pain', 'acute_pain', 'chronic_pain']:
        print(f'Saved AL/plots/model_parameter_by_pain_{group}.png (+ .svg)')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Re-create only the parameter x pain severity plots.')
    parser.add_argument('--file_path', default=DATA_FILE_PATH, help='Folder with the raw data files (default: as in AL_main.py)')
    parser.add_argument('--file_names', nargs='+', default=DATA_FILE_NAMES, help='Raw data file names (default: as in AL_main.py)')
    parser.add_argument('--model', default=None, help='Model to plot, e.g. Advantage+novel (default: best model by BIC)')
    args = parser.parse_args()

    main(args.file_path, args.file_names, args.model)
