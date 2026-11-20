import os
import sys
import argparse
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from helpers.pipeline import Pipeline

"""
Lite script: transfer phase choice rates split into the EARLY and LATE halves of the task.

The halves split the task, not the data: each participant's transfer trials are ordered by trial_index
and the first half of ALL their transfer trials (total count / 2, counted before any trials are removed)
is 'early', the rest 'late'. Trials are then processed and excluded exactly as in the pipeline
(pain cutoff, accuracy exclusion, RT exclusion, same-value pairs removed), so each half keeps
whatever valid trials fall within it.

Outputs (AL/stats/):
    transfer_halves_choice_rates.csv        - choice rate per participant x stimulus type x half
    transfer_halves_summary.csv             - mean (SD) choice rate per group x stimulus type x half
    transfer_halves_stimulus_changes.csv    - early vs late paired t-tests per group x stimulus type (Holm-corrected across all 15)
    transfer_halves_planned_comparisons.csv - Table 2 group x stimulus type planned comparisons, per half
    transfer_halves_glmm_n_chosen_results.csv  - GLMM: group x stimulus type x half (does the group pattern differ between halves?)
    transfer_halves_difference_changes.csv  - paired t-tests early vs late of HR-LP and LR-LP, within each group
    transfer_halves_difference_glmm_*_results.csv - LMMs: group x half on the HR-LP and LR-LP differences

Statistical tests (R is required, as for the main pipeline):
    1. Binomial GLMM of choice counts (Equation 3 plus half):
       cbind(n_chosen, n_ignored) ~ group x stimulus type x half + (1 + stimulus type x half | participant)
       A non-significant group x stimulus type x half interaction indicates the group pattern did not differ between halves.
    2. For the key pair-specific differences (HR-LP, LR-LP): paired t-tests early vs late within each group, and an LMM
       difference ~ group x half + (1 | participant), where the group x half interaction tests whether group differences changed.

Run from the repository root (same as AL_main.py):
    python AL/lite_transfer_halves.py
"""

#Same settings as AL_main.py
KWARGS = {'file_path':                      r'AL/data',
          'file_name':                      [r'v1a_avoid_pain.csv', r'v1b_avoid_paindepression.csv'],
          'accuracy_exclusion_threshold':   70,
          'RT_low_threshold':               200,
          'RT_high_threshold':              5000,
          'rscripts_path':                  'C:/Program Files/R/R-4.5.0/bin/x64/Rscript',
          'load_stats':                     True}

SYMBOLS = {4: 'High Reward', 3: 'Low Reward', 2: 'Low Punish', 1: 'High Punish', 0: 'Novel'}
SYMBOL_ORDER = ['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel']
HALVES = ['early', 'late']
GROUP_COMPARISONS = [['no pain', 'acute pain'], ['no pain', 'chronic pain'], ['acute pain', 'chronic pain']]


def assign_halves(raw_data: pd.DataFrame) -> pd.DataFrame:

    """
    Label each participant's transfer trials as early/late using their full transfer trial count
    (before any removals): positions 1..n/2 are early, the rest late.
    """

    transfer = raw_data[raw_data['trial_type'] == 'probe'][['participant_id', 'trial_index']].copy()
    transfer = transfer.sort_values(['participant_id', 'trial_index'])
    position = transfer.groupby('participant_id').cumcount() + 1
    n_trials = transfer.groupby('participant_id')['trial_index'].transform('size')
    transfer['half'] = np.where(position <= n_trials / 2, 'early', 'late')
    return transfer


def choice_rates(data: pd.DataFrame, group_code: str) -> pd.DataFrame:

    """
    Choice rate (%) = times chosen / times presented, per participant x stimulus type (as in compute_choice_rate)
    """

    chosen = data.groupby([group_code, 'participant_id', 'symbol_chosen']).size()
    ignored = data.groupby([group_code, 'participant_id', 'symbol_ignored']).size()
    chosen.index.names = ignored.index.names = [group_code, 'participant_id', 'symbol']
    counts = pd.concat({'n_chosen': chosen, 'n_ignored': ignored}, axis=1).fillna(0)
    counts['choice_rate'] = counts['n_chosen'] / (counts['n_chosen'] + counts['n_ignored']) * 100
    counts = counts.reset_index()
    counts['symbol'] = counts['symbol'].map(SYMBOLS)
    return counts


def pair_differences(data: pd.DataFrame, group_code: str, symbol: int, versus: int = 2) -> pd.DataFrame:

    """
    Pair-specific choice rate difference (as in Table 2's group x stimulus type planned comparisons):
    on trials presenting exactly this pair, rate of choosing `symbol` minus rate of choosing `versus`.
    """

    pair = data[((data['symbol_chosen'] == symbol) & (data['symbol_ignored'] == versus)) |
                ((data['symbol_chosen'] == versus) & (data['symbol_ignored'] == symbol))]
    rate = pair.groupby([group_code, 'participant_id'])['symbol_chosen'].apply(lambda x: (x == symbol).mean() * 100)
    difference = (rate - (100 - rate)).rename('choice_rate').reset_index()
    return difference


def main(kwargs: dict) -> None:

    pipeline = Pipeline()
    pipeline.set_parameters(**kwargs)
    group_code = pipeline.group_code

    #Load data and label the halves on the raw transfer trials (before any removals)
    pipeline.load_data(file_path=pipeline.file_path, file_name=pipeline.file_name)
    halves = assign_halves(pipeline.data)

    #Process the data exactly as the pipeline does
    pipeline.exclude_pain(threshold=pipeline.pain_cutoff)
    pipeline.process_data()
    transfer = pipeline.transfer_data.merge(halves, on=['participant_id', 'trial_index'], how='left')
    if transfer['half'].isna().any():
        raise ValueError('Some processed transfer trials could not be matched to a half.')

    #Sanity check: combining both halves reproduces the pipeline's choice rates
    full = choice_rates(transfer, group_code).merge(pipeline.choice_rate.reset_index(), on=[group_code, 'participant_id', 'symbol'], suffixes=('', '_pipeline'))
    max_difference = (full['choice_rate'] - full['choice_rate_pipeline'].astype(float)).abs().max()
    print(f'Check against pipeline choice rates (should be ~0): {max_difference:.2e}')

    #Choice rates per half
    rates = pd.concat([choice_rates(transfer[transfer['half'] == half], group_code).assign(half=half) for half in HALVES])
    rates.to_csv('AL/stats/transfer_halves_choice_rates.csv', index=False)

    trials = transfer.groupby(['participant_id', 'half']).size().unstack()
    print(f"\nValid trials per participant: early mean {trials['early'].mean():.1f} (min {trials['early'].min()}), "
          f"late mean {trials['late'].mean():.1f} (min {trials['late'].min()})")

    #Mean (SD) per group x stimulus type x half
    summary = rates.groupby([group_code, 'half', 'symbol'])['choice_rate'].agg(['mean', 'std', 'count']).reset_index()
    summary.to_csv('AL/stats/transfer_halves_summary.csv', index=False)
    summary['M (SD)'] = summary.apply(lambda row: f"{row['mean']:.2f} ({row['std']:.2f})", axis=1)
    table = summary.pivot(index='symbol', columns=[group_code, 'half'], values='M (SD)')
    table = table.reindex(SYMBOL_ORDER)[[(group, half) for group in pipeline.group_labels for half in HALVES]]
    pd.set_option('display.width', 250)
    print('\nChoice rate (%), mean (SD):')
    print(table.to_string())

    #Early vs late within each group x stimulus type: paired t-tests, Holm-corrected across all tests
    stimulus_changes = stimulus_change_tests(rates, group_code, pipeline.group_labels)
    stimulus_changes.to_csv('AL/stats/transfer_halves_stimulus_changes.csv', index=False)
    stimulus_changes['cell'] = stimulus_changes.apply(lambda row: f"{row['early']} -> {row['late']}, p = {format_p(row['p_holm'])}", axis=1)
    change_table = stimulus_changes.pivot(index='symbol', columns='group', values='cell').reindex(SYMBOL_ORDER)[list(pipeline.group_labels)]
    print('\nWithin stimulus type, early M (SD) -> late M (SD), paired t-test p (Holm-corrected across all tests):')
    print(change_table.to_string())

    #Table 2 planned comparisons (pair-specific HR-LP and LR-LP), separately for each half
    planned = []
    for half in HALVES:
        half_data = transfer[transfer['half'] == half]
        for symbol, label in [(4, 'HR-LP'), (3, 'LR-LP')]:
            differences = pair_differences(half_data, group_code, symbol)
            result = pipeline.planned_ttests('choice_rate', group_code, GROUP_COMPARISONS, differences)['model_summary']
            planned.append(result.assign(half=half, difference=label))
    planned = pd.concat(planned)[['half', 'difference', 'comparison', 'df', 't_value', 'p_value', 'cohens_d']]
    planned.to_csv('AL/stats/transfer_halves_planned_comparisons.csv', index=False)
    print('\nGroup x stimulus type planned comparisons (pair-specific, as in Table 2), per half:')
    print(planned.round(4).to_string(index=False))

    #----- Test 1: GLMM of choice counts with half (Equation 3 + half) -----
    counts = rates[[group_code, 'participant_id', 'symbol', 'half', 'n_chosen', 'n_ignored']].copy()
    counts[['n_chosen', 'n_ignored']] = counts[['n_chosen', 'n_ignored']].astype(int)
    counts.to_csv('AL/stats/transfer_halves_counts.csv', index=False)
    formula = f'cbind(n_chosen,n_ignored)~1+{group_code}*symbol*half+(1+symbol*half|participant_id)'
    glmm = pipeline.generalized_linear_model(formula, counts,
                                             path=pipeline.repo_directory,
                                             filename='AL/stats/transfer_halves_counts.csv',
                                             savename='AL/stats/transfer_halves_glmm.csv',
                                             family='binomial',
                                             method='Wald')
    print('\nTest 1. GLMM of choice counts: group x stimulus type x half (Type III Wald chi-square):')
    print(format_glmm(glmm))

    #----- Test 2: key pair-specific differences early vs late -----
    differences = pd.concat([pair_differences(transfer[transfer['half'] == half], group_code, symbol).assign(half=half, difference=label)
                             for half in HALVES for symbol, label in [(4, 'HR-LP'), (3, 'LR-LP')]])

    #2a. Paired t-tests early vs late, within each group
    changes = []
    for label in ['HR-LP', 'LR-LP']:
        for group in pipeline.group_labels:
            group_data = differences[(differences['difference'] == label) & (differences[group_code] == group)]
            group_data = group_data[group_data.groupby('participant_id')['half'].transform('nunique') == 2] #participants with both halves
            result = pipeline.planned_ttests('choice_rate', 'half', [['early', 'late']], group_data)['model_summary']
            changes.append(result.assign(difference=label, group=group))
    changes = pd.concat(changes)[['difference', 'group', 'comparison', 'df', 't_value', 'p_value', 'cohens_d']]
    changes.to_csv('AL/stats/transfer_halves_difference_changes.csv', index=False)
    print('\nTest 2a. Early vs late, paired t-tests within each group:')
    print(changes.round(4).to_string(index=False))

    #2b. LMM: does the group difference change between halves?
    print('\nTest 2b. LMM difference ~ group x half + (1 | participant) (Type III Wald chi-square):')
    for label in ['HR-LP', 'LR-LP']:
        tag = label.replace('-', '_')
        label_data = differences[differences['difference'] == label][[group_code, 'participant_id', 'half', 'choice_rate']]
        label_data.to_csv(f'AL/stats/transfer_halves_difference_{tag}.csv', index=False)
        lmm = pipeline.generalized_linear_model(f'choice_rate~1+{group_code}*half+(1|participant_id)', label_data,
                                                path=pipeline.repo_directory,
                                                filename=f'AL/stats/transfer_halves_difference_{tag}.csv',
                                                savename=f'AL/stats/transfer_halves_difference_glmm_{tag}.csv',
                                                family='gaussian')
        print(f'{label}:')
        print(format_glmm(lmm))


def stimulus_change_tests(rates: pd.DataFrame, group_code: str, groups: list) -> pd.DataFrame:

    """
    Early vs late paired t-tests of choice rates within each group x stimulus type,
    with Holm correction across all group x stimulus type tests
    """

    wide = rates.pivot_table(index=[group_code, 'participant_id', 'symbol'], columns='half', values='choice_rate').dropna().reset_index()
    rows = []
    for group in groups:
        for symbol in SYMBOL_ORDER:
            cell = wide[(wide[group_code] == group) & (wide['symbol'] == symbol)]
            ttest = stats.ttest_rel(cell['early'], cell['late'])
            change = cell['early'] - cell['late']
            rows.append({'group': group,
                         'symbol': symbol,
                         'early': f"{cell['early'].mean():.2f} ({cell['early'].std():.2f})",
                         'late': f"{cell['late'].mean():.2f} ({cell['late'].std():.2f})",
                         'early_mean': cell['early'].mean(), 'early_sd': cell['early'].std(),
                         'late_mean': cell['late'].mean(), 'late_sd': cell['late'].std(),
                         't_value': ttest.statistic,
                         'df': len(cell) - 1,
                         'p_value': ttest.pvalue,
                         'cohens_d': change.mean() / change.std(ddof=1)})
    results = pd.DataFrame(rows)
    results['p_holm'] = multipletests(results['p_value'], method='holm')[1]
    return results


def format_p(p: float) -> str:

    """
    Format a p-value to four decimals without a leading zero ('<.0001' below that)
    """

    return '<.0001' if p < .0001 else f'{p:.4f}'.replace('0.', '.', 1)


def format_glmm(model: dict) -> str:

    """
    Format a GLMM summary from the pipeline as effect, chi-square, df, p
    """

    summary = model['model_summary'].rename(columns={'factor': 'effect', 'test_value': 'chi_square'})
    return summary[['effect', 'chi_square', 'df', 'p_value']].round(4).to_string(index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer choice rates in the early and late halves of the transfer phase.')
    parser.add_argument('--file_path', default=KWARGS['file_path'], help='Folder with the raw data files (default: as in AL_main.py)')
    parser.add_argument('--file_names', nargs='+', default=KWARGS['file_name'], help='Raw data file names (default: as in AL_main.py)')
    parser.add_argument('--rscripts_path', default=KWARGS['rscripts_path'], help='Path to Rscript (default: as in AL_main.py)')
    args = parser.parse_args()

    os.makedirs('AL/stats', exist_ok=True)
    main({**KWARGS, 'file_path': args.file_path, 'file_name': args.file_names, 'rscripts_path': args.rscripts_path})
