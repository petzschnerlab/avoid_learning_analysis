import os
import sys
import pickle
import argparse
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from helpers.plotting import Plotting
from helpers.report_functions import ReportFunctions

"""
Lite script: re-creates ONLY the model parameters by pain duration plot (one parameter per row; learning rates log-transformed),
without running the full AL pipeline. Uses the same plotting function as the pipeline (Plotting.plot_parameters_by_duration).

Run from the repository root (same as AL_main.py):
    python AL/lite_parameter_duration_plot.py

Inputs (must already exist from a previous full run / the modelling package):
    AL/modelling/fit_data_FIT.pkl   - fitted parameters per model
    AL/modelling/group_BIC.csv      - used to pick the best model (same rule as the pipeline)
    The raw data files listed in DATA_FILE_NAMES - used for the pain duration

Outputs:
    AL/plots/parameter-by-duration.png / .svg   (same file as the pipeline/report)
"""

#Same data settings as AL_main.py
DATA_FILE_PATH = r'AL/data'
DATA_FILE_NAMES = [r'v1a_avoid_pain.csv', r'v1b_avoid_paindepression.csv']

#Duration coding, as in ReportFunctions.load_modelling_results
DURATION_CODES = {'I am not in pain': 0, '< 2 weeks': 1, '2-4 weeks': 2, '1 – 3 months': 3, '3 – 6 months': 4,
                  '6 – 12 months': 5, '1 – 5 years': 6, '> 5 years': 7, '> 10 years': 8}


def get_best_model() -> str:

    """Select the best model exactly as the pipeline does: the lowest BIC in the last row ('full') of group_BIC.csv"""

    model_BIC = pd.read_csv('AL/modelling/group_BIC.csv', index_col=0)
    model_BIC = model_BIC.iloc[:, :-1].round().astype(int) #drop the stored best_model column
    return model_BIC.idxmin(axis=1).values[-1]


def load_durations(file_path: str, file_names: list) -> pd.DataFrame:

    """One pain duration per participant, coded 0-8 as in the pipeline"""

    data = pd.concat([pd.read_csv(os.path.join(file_path, file_name)) for file_name in file_names], axis=0)
    durations = data.groupby('participant_id')['duration'].first().reset_index()
    durations['pain_duration'] = durations['duration'].map(DURATION_CODES)
    unknown = durations.loc[durations['pain_duration'].isna() & durations['duration'].notna(), 'duration'].unique()
    if len(unknown) > 0:
        warnings.warn(f'Unrecognised duration labels (dropped): {list(unknown)}')
    return durations[['participant_id', 'pain_duration']]


def main(file_path: str, file_names: list, model: str = None) -> None:

    best_model = model if model is not None else get_best_model()
    print(f'Model: {best_model}')

    with open('AL/modelling/fit_data_FIT.pkl', 'rb') as f:
        fit_data = pickle.load(f)[best_model]
    fit_data = fit_data.rename(columns={'participant': 'participant_id'})
    parameter_names = list(fit_data.columns[4:])
    print(f'Parameters: {parameter_names}')

    fit_data = fit_data.merge(load_durations(file_path, file_names), on='participant_id', how='left')
    n_missing = fit_data['pain_duration'].isna().sum()
    if n_missing > 0:
        warnings.warn(f'{n_missing} fitted participant(s) have no usable pain duration and are not plotted.')

    os.makedirs('AL/plots', exist_ok=True)
    #Log-transform learning rates (same functions the pipeline uses)
    report_functions = ReportFunctions()
    log_parameters = report_functions.get_learning_rate_parameters(parameter_names)
    print(f'Log-transformed: {log_parameters}')
    plot_data = report_functions.log_transform_parameters(fit_data, log_parameters)
    Plotting().plot_parameters_by_duration(plot_data, parameter_names, save_name='parameter-by-duration', log_parameters=log_parameters)
    print('Saved AL/plots/parameter-by-duration.png (+ .svg)')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Re-create only the parameters by pain duration plot.')
    parser.add_argument('--file_path', default=DATA_FILE_PATH, help='Folder with the raw data files (default: as in AL_main.py)')
    parser.add_argument('--file_names', nargs='+', default=DATA_FILE_NAMES, help='Raw data file names (default: as in AL_main.py)')
    parser.add_argument('--model', default=None, help='Model to use, e.g. Advantage+novel (default: best model by BIC)')
    args = parser.parse_args()

    main(args.file_path, args.file_names, args.model)
