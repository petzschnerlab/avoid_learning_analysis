import sys
import argparse
import warnings
sys.dont_write_bytecode = True
warnings.simplefilter(action='ignore', category=FutureWarning)

from helpers.pipeline import Pipeline

"""
Lite script: re-creates ONLY the transfer-phase choice matrix figure
(AL/plots/pain/transfer-choice-matrix.png/.svg, plus the group-mean values as .csv),
without running the full AL pipeline (no statistics, no R, no report).

Each cell is the percentage of trials on which the row stimulus type was chosen when it was
presented with the column stimulus type (only trials of that exact pairing), for each group.

Run from the repository root (same as AL_main.py):
    python AL/lite_transfer_choice_matrix.py
"""

#Same settings as AL_main.py
KWARGS = {'file_path':                      r'AL/data',
          'file_name':                      [r'v1a_avoid_pain.csv', r'v1b_avoid_paindepression.csv'],
          'accuracy_exclusion_threshold':   70,
          'RT_low_threshold':               200,
          'RT_high_threshold':              5000}


def main(kwargs: dict) -> None:

    pipeline = Pipeline()
    pipeline.set_parameters(**kwargs)

    #Load and process the data (same steps as Pipeline.run)
    pipeline.load_data(file_path=pipeline.file_path, file_name=pipeline.file_name)
    pipeline.exclude_pain(threshold=pipeline.pain_cutoff)
    pipeline.process_data()

    #Plot
    pipeline.plot_transfer_choice_matrix('transfer-choice-matrix')
    print(f'Saved AL/plots/{pipeline.split_by_group}/transfer-choice-matrix.png (+ .svg, .csv)')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Re-create only the transfer choice matrix figure.')
    parser.add_argument('--file_path', default=KWARGS['file_path'], help='Folder with the raw data files (default: as in AL_main.py)')
    parser.add_argument('--file_names', nargs='+', default=KWARGS['file_name'], help='Raw data file names (default: as in AL_main.py)')
    args = parser.parse_args()

    main({**KWARGS, 'file_path': args.file_path, 'file_name': args.file_names})
