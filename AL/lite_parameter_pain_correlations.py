import os
import sys
import pickle
import argparse
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

"""
Lite script: prints the parameter x pain severity Spearman correlations (Table S9), plus the same analysis
with the acute and chronic pain groups combined, without running the full AL pipeline.

Same computations as ReportFunctions.load_modelling_results:
    Separate groups (Table S9): Spearman rho for each parameter x pain metric within acute and chronic pain;
        p-values for the composite score only; Holm correction across the 10 composite tests
        (5 parameters x 2 groups).
    Combined pain group: acute + chronic pain participants pooled; Spearman rho for each parameter x pain metric;
        p-values for the composite score only; Holm correction across its own 5 composite tests.
    Pain duration: Spearman rho between each parameter and pain duration (coded 1-8 for the pain groups, as in the
        pipeline), for the combined group (Holm across 5 tests) and for acute and chronic pain separately (Holm across 10).

Run from the repository root (same as AL_main.py):
    python AL/lite_parameter_pain_correlations.py

Inputs (must already exist from a previous full run / the modelling package):
    AL/modelling/fit_data_FIT.pkl   - fitted parameters per model (includes pain_group)
    AL/modelling/group_BIC.csv      - used to pick the best model (same rule as the pipeline)
    The raw data files listed in DATA_FILE_NAMES - used for the pain scores

Outputs (AL/stats/):
    parameter_pain_correlations_groups.csv    - Table S9 (acute and chronic pain separately)
    parameter_pain_correlations_combined.csv  - acute and chronic pain combined
    parameter_duration_correlations.csv       - parameter x pain duration (combined, acute, chronic)
"""

#Same data settings as AL_main.py
DATA_FILE_PATH = r'AL/data'
DATA_FILE_NAMES = [r'v1a_avoid_pain.csv', r'v1b_avoid_paindepression.csv']
PAIN_NAMES = ['intensity', 'unpleasant', 'interference', 'composite']

#Duration coding, as in ReportFunctions.load_modelling_results (ordinal: 0 = not in pain ... 8 = > 10 years)
DURATION_CODES = {'I am not in pain': 0, '< 2 weeks': 1, '2-4 weeks': 2, '1 – 3 months': 3, '3 – 6 months': 4,
                  '6 – 12 months': 5, '1 – 5 years': 6, '> 5 years': 7, '> 10 years': 8}


def get_best_model() -> str:

    """Select the best model exactly as the pipeline does: the lowest BIC in the last row ('full') of group_BIC.csv"""

    model_BIC = pd.read_csv('AL/modelling/group_BIC.csv', index_col=0)
    model_BIC = model_BIC.iloc[:, :-1].round().astype(int) #drop the stored best_model column
    return model_BIC.idxmin(axis=1).values[-1]


def load_pain_scores(file_path: str, file_names: list) -> pd.DataFrame:

    """One pain score per participant, rescaled to 0-10 as in Processing.load_data, plus the composite score and pain duration (0-8)"""

    data = pd.concat([pd.read_csv(os.path.join(file_path, file_name)) for file_name in file_names], axis=0)
    pain_scores = data.groupby('participant_id')[['intensity', 'unpleasant', 'interference', 'duration']].first().reset_index()
    pain_scores[['intensity', 'unpleasant', 'interference']] = pain_scores[['intensity', 'unpleasant', 'interference']] / 10
    pain_scores['composite'] = pain_scores[['intensity', 'unpleasant', 'interference']].mean(axis=1)
    pain_scores['pain_duration'] = pain_scores['duration'].map(DURATION_CODES)
    unknown = pain_scores.loc[pain_scores['pain_duration'].isna() & pain_scores['duration'].notna(), 'duration'].unique()
    if len(unknown) > 0:
        warnings.warn(f'Unrecognised duration labels (treated as missing): {list(unknown)}')
    return pain_scores.drop(columns='duration')


def duration_rows(data: pd.DataFrame, label: str, parameter_names: list) -> list:

    """Spearman rho between each parameter and pain duration (uncorrected p)"""

    rows = []
    for parameter in parameter_names:
        d = data[[parameter, 'pain_duration']].dropna()
        r, p = stats.spearmanr(d[parameter], d['pain_duration'])
        rows.append({'group': label, 'parameter': parameter, 'n': len(d), 'rho': r, 'p': p})
    return rows


def format_rho(r: float) -> str:
    return 'NA' if np.isnan(r) else f'{r:.2f}'.replace('0.', '.')


def format_p(p: float) -> str:
    return 'NA' if np.isnan(p) else ('< .0001' if p < .0001 else f'{p:.4f}'.replace('0.', '.'))


def correlation_rows(data: pd.DataFrame, label: str, parameter_names: list) -> list:

    """Spearman rho for each parameter x pain metric, with the composite p-value (uncorrected)"""

    rows = []
    for parameter in parameter_names:
        row = {'group': label, 'parameter': parameter, 'n': int(data[[parameter, 'composite']].dropna().shape[0])}
        for pain_metric in PAIN_NAMES:
            d = data[[parameter, pain_metric]].dropna()
            r, p = stats.spearmanr(d[parameter], d[pain_metric])
            row[pain_metric] = r
            if pain_metric == 'composite':
                row['p'] = p
        rows.append(row)
    return rows


def print_table(table: pd.DataFrame, title: str) -> None:

    display = table.copy()
    for column in PAIN_NAMES:
        display[column] = display[column].apply(format_rho)
    display['p'] = display['p'].apply(format_p)
    display['p_Holm'] = display['p_Holm'].apply(format_p)
    print(f'\n{title}')
    print(display.set_index(['group', 'parameter']).to_string())


def main(file_path: str, file_names: list, model: str = None) -> None:

    best_model = model if model is not None else get_best_model()
    print(f'Model: {best_model}')

    #Load the fitted parameters and merge in pain scores
    with open('AL/modelling/fit_data_FIT.pkl', 'rb') as f:
        fit_data = pickle.load(f)[best_model]
    fit_data = fit_data.rename(columns={'participant': 'participant_id'})
    parameter_names = list(fit_data.columns[4:])
    fit_data = fit_data.merge(load_pain_scores(file_path, file_names), on='participant_id', how='left')

    n_missing = fit_data['composite'].isna().sum()
    if n_missing > 0:
        warnings.warn(f'{n_missing} fitted participant(s) have no pain scores in the data files.')

    #Separate groups (Table S9): Holm across the 10 composite tests
    rows = []
    for group in ['acute pain', 'chronic pain']:
        rows.extend(correlation_rows(fit_data[fit_data['pain_group'] == group], group, parameter_names))
    groups = pd.DataFrame(rows)
    groups['p_Holm'] = multipletests(groups['p'], method='holm')[1]

    #Combined pain group: Holm across its 5 composite tests
    combined_data = fit_data[fit_data['pain_group'].isin(['acute pain', 'chronic pain'])]
    combined = pd.DataFrame(correlation_rows(combined_data, 'acute and chronic pain', parameter_names))
    combined['p_Holm'] = multipletests(combined['p'], method='holm')[1]

    os.makedirs('AL/stats', exist_ok=True)
    groups.to_csv('AL/stats/parameter_pain_correlations_groups.csv', index=False)
    combined.to_csv('AL/stats/parameter_pain_correlations_combined.csv', index=False)

    #Pain duration: same groups and Holm families as above (5 tests combined; 10 tests across acute and chronic)
    duration_combined = pd.DataFrame(duration_rows(combined_data, 'acute and chronic pain', parameter_names))
    duration_combined['p_Holm'] = multipletests(duration_combined['p'], method='holm')[1]
    duration_groups = []
    for group in ['acute pain', 'chronic pain']:
        duration_groups.extend(duration_rows(fit_data[fit_data['pain_group'] == group], group, parameter_names))
    duration_groups = pd.DataFrame(duration_groups)
    duration_groups['p_Holm'] = multipletests(duration_groups['p'], method='holm')[1]
    duration = pd.concat([duration_combined, duration_groups], ignore_index=True)
    duration.to_csv('AL/stats/parameter_duration_correlations.csv', index=False)

    pd.set_option('display.width', 250)
    print_table(groups, 'Spearman rho, acute and chronic pain separately (Table S9; Holm across 10 composite tests)')
    print_table(combined, 'Spearman rho, acute and chronic pain combined (Holm across 5 composite tests)')
    display = duration.copy()
    display['rho'] = display['rho'].apply(format_rho)
    display['p'] = display['p'].apply(format_p)
    display['p_Holm'] = display['p_Holm'].apply(format_p)
    print('\nSpearman rho with pain duration (Holm across 5 tests for combined; across 10 tests for acute and chronic)')
    print(display.set_index(['group', 'parameter']).to_string())
    print('\nn: ' + ', '.join(f"{g} = {fit_data[fit_data['pain_group'] == g]['participant_id'].nunique()}" for g in ['acute pain', 'chronic pain'])
          + f", combined = {combined_data['participant_id'].nunique()}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Print the parameter x pain severity correlations, including the combined pain group.')
    parser.add_argument('--file_path', default=DATA_FILE_PATH, help='Folder with the raw data files (default: as in AL_main.py)')
    parser.add_argument('--file_names', nargs='+', default=DATA_FILE_NAMES, help='Raw data file names (default: as in AL_main.py)')
    parser.add_argument('--model', default=None, help='Model to use, e.g. Advantage+novel (default: best model by BIC)')
    args = parser.parse_args()

    main(args.file_path, args.file_names, args.model)
