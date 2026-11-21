import os
import sys
import argparse
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from helpers.pipeline import Pipeline

"""
Lite script: prints transfer_rt_posthoc_interaction (Tukey HSD post hocs for transfer RT across all
group x stimulus type combinations), exactly as computed in Statistics.run_statistics:

    factors = [self.group_code, 'symbol']
    self.transfer_rt_posthoc_interaction = self.post_hoc_tests('choice_rt', factors, self.choice_rt.reset_index())

It also prints mean (SD) transfer RT per group x stimulus type, and the subset of post hocs that compare
groups within the same stimulus type (the comparisons that unpack the group x stimulus type interaction).

Outputs (AL/stats/):
    transfer_rt_posthoc_interaction.csv            - all Tukey HSD comparisons
    transfer_rt_posthoc_interaction_within_stimulus.csv - group comparisons within each stimulus type
    transfer_rt_means.csv                          - mean (SD) RT per group x stimulus type
    transfer_rt_hr_advantage.csv                   - per-participant RT differences relative to HR
    transfer_rt_hr_advantage_tests.csv             - group comparisons of those differences (Holm across 12)

HR advantage analysis (follow-up of the group x stimulus type interaction):
    For each participant, RT(stimulus) - RT(HR) for LR, LP, HP and N (how much slower than HR).
    Overall advantage = mean of the four differences, compared across groups with a one-way ANOVA.
    Each of the four differences is compared between groups (no vs acute, no vs chronic, acute vs chronic)
    with the pipeline's planned t-tests (Welch-corrected when variances differ), Holm-corrected across all 12.

Run from the repository root (same as AL_main.py):
    python AL/lite_transfer_rt_posthoc.py
"""

#Same settings as AL_main.py
KWARGS = {'file_path':                      r'AL/data',
          'file_name':                      [r'v1a_avoid_pain.csv', r'v1b_avoid_paindepression.csv'],
          'accuracy_exclusion_threshold':   70,
          'RT_low_threshold':               200,
          'RT_high_threshold':              5000}

SYMBOL_ORDER = ['High Reward', 'Low Reward', 'Low Punish', 'High Punish', 'Novel']


def main(kwargs: dict) -> None:

    pipeline = Pipeline()
    pipeline.set_parameters(**kwargs)
    group_code = pipeline.group_code

    #Load and process the data (same steps as Pipeline.run)
    pipeline.load_data(file_path=pipeline.file_path, file_name=pipeline.file_name)
    pipeline.exclude_pain(threshold=pipeline.pain_cutoff)
    pipeline.process_data()

    choice_rt = pipeline.choice_rt.reset_index()
    choice_rt['choice_rt'] = choice_rt['choice_rt'].astype(float)

    #Mean (SD) RT per group x stimulus type
    means = choice_rt.groupby([group_code, 'symbol'])['choice_rt'].agg(['mean', 'std', 'count']).reset_index()
    means.to_csv('AL/stats/transfer_rt_means.csv', index=False)
    means['M (SD)'] = means.apply(lambda row: f"{row['mean']:.1f} ({row['std']:.1f})", axis=1)
    mean_table = means.pivot(index='symbol', columns=group_code, values='M (SD)').reindex(SYMBOL_ORDER)[list(pipeline.group_labels)]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    print('\nTransfer RT (ms), mean (SD):')
    print(mean_table.to_string())

    #Tukey HSD post hocs across all group x stimulus type combinations (as in run_statistics)
    factors = [group_code, 'symbol']
    posthoc = pipeline.post_hoc_tests('choice_rt', factors, choice_rt.copy())
    posthoc.to_csv('AL/stats/transfer_rt_posthoc_interaction.csv')
    print(f'\ntransfer_rt_posthoc_interaction: all {len(posthoc)} Tukey HSD comparisons:')
    print(posthoc.to_string())

    #Group comparisons within the same stimulus type (factor labels are "group & symbol vs group & symbol")
    def same_stimulus(comparison: str) -> bool:
        left, right = comparison.split(' vs ')
        return left.split(' & ')[1] == right.split(' & ')[1]

    within = posthoc[[same_stimulus(comparison) for comparison in posthoc.index]]
    within.to_csv('AL/stats/transfer_rt_posthoc_interaction_within_stimulus.csv')
    print(f'\nGroup comparisons within each stimulus type ({len(within)} of the {len(posthoc)}; Tukey-adjusted across all {len(posthoc)}):')
    print(within.to_string())

    #HR advantage: how much slower each stimulus is chosen than HR, per participant
    wide = choice_rt.pivot_table(index=[group_code, 'participant_id'], columns='symbol', values='choice_rt').reset_index()
    others = ['Low Reward', 'Low Punish', 'High Punish', 'Novel']
    for symbol in others:
        wide[f'{symbol} - High Reward'] = wide[symbol] - wide['High Reward']
    wide['overall advantage'] = wide[[f'{symbol} - High Reward' for symbol in others]].mean(axis=1)
    wide.to_csv('AL/stats/transfer_rt_hr_advantage.csv', index=False)

    groups = list(pipeline.group_labels)
    comparisons = [['no pain', 'acute pain'], ['no pain', 'chronic pain'], ['acute pain', 'chronic pain']]

    print('\nHR advantage: RT(stimulus) - RT(HR), ms, mean (SD):')
    columns = ['overall advantage'] + [f'{symbol} - High Reward' for symbol in others]
    advantage_table = pd.DataFrame({group: [f"{wide[wide[group_code] == group][column].mean():.1f} ({wide[wide[group_code] == group][column].std():.1f})" for column in columns] for group in groups}, index=columns)
    print(advantage_table.to_string())

    overall = [wide[wide[group_code] == group]['overall advantage'].dropna() for group in groups]
    anova = stats.f_oneway(*overall)
    print(f"\nOverall HR advantage, one-way ANOVA: F(2, {sum(len(x) for x in overall) - 3}) = {anova.statistic:.2f}, p = {anova.pvalue:.4f}")

    tests = []
    for symbol in others:
        column = f'{symbol} - High Reward'
        data = wide.dropna(subset=[column]).rename(columns={column: 'difference'})
        result = pipeline.planned_ttests('difference', group_code, comparisons, data)['model_summary']
        tests.append(result.assign(stimulus=symbol))
    tests = pd.concat(tests)
    tests['p_holm'] = multipletests(tests['p_value'], method='holm')[1]
    tests = tests[['stimulus', 'comparison', 'df', 't_value', 'p_value', 'p_holm', 'cohens_d', 'homogeneity_assumption']]
    tests.to_csv('AL/stats/transfer_rt_hr_advantage_tests.csv', index=False)
    print('\nGroup comparisons of RT(stimulus) - RT(HR), Holm-corrected across all 12:')
    print(tests.round(4).to_string(index=False))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Print the transfer RT group x stimulus type post hocs.')
    parser.add_argument('--file_path', default=KWARGS['file_path'], help='Folder with the raw data files (default: as in AL_main.py)')
    parser.add_argument('--file_names', nargs='+', default=KWARGS['file_name'], help='Raw data file names (default: as in AL_main.py)')
    args = parser.parse_args()

    os.makedirs('AL/stats', exist_ok=True)
    main({**KWARGS, 'file_path': args.file_path, 'file_name': args.file_names})
