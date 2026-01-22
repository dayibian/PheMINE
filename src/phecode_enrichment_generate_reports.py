# This script is used to generate reports for the enrichment analysis results
# It will generate a table of enriched phecodes with statistics

import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import argparse
import sys
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import yaml
# Load config.yaml for default paths
try:
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Try to open config.yaml in the current directory
    with open("config.yaml") as f:
        config = yaml.safe_load(f)


def setup_log(fn_log, mode='w'):
    '''
    Print log message to console and write to a log file.
    Will overwrite existing log file by default
    Params:
    - fn_log: name of the log file
    - mode: writing mode. Change mode='a' for appending
    '''
    # f string is not fully compatible with logging, so use %s for string formatting
    logging.root.handlers = [] # Remove potential handler set up by others (especially in google colab)
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(filename=fn_log, mode=mode),
                                  logging.StreamHandler()], format='%(message)s')

def process_args():
    '''
    Process arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='High-level folder that contains data for each trait', type=str,
                        default='../data/')
    parser.add_argument('--output_folder', help='High-level folder that contains output for each trait', type=str,
                        default='../results/')
    parser.add_argument('--trait', help='Trait of interest', type=str, default='als')
    parser.add_argument('--input_prefix', help='The prefix for the input file.', type=str, default='output')
    parser.add_argument('--phecode_map_file', type=str, default=config['phecode_map_file'], help='Path to the phecode map file')
    
    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(args.output_folder) / f'{args.trait}_{args.input_prefix}_report.log'
    setup_log(fn_log, mode='a')

    # Record script used
    cmd_used = 'python ' + ' '.join(sys.argv)

    logging.info('\n# Call used:')
    logging.info(cmd_used+'\n')
    
    logging.info('# Arguments used:')
    for arg in vars(args):
        cmd_used += f' --{arg} {getattr(args, arg)}'
        msg = f'# - {arg}: {getattr(args, arg)}'
        logging.info(msg)

    return args


def main():
    args = process_args()
    TRAIT = args.trait
    data_path = Path(args.data_folder)
    output_path = Path(args.output_folder)
    prefix = args.input_prefix

    logging.info('\nReading enrichment analysis results...')
    results = pd.read_csv(output_path / f'{prefix}.counts_and_pval.txt', sep='\t', dtype={'phecode':str})
    results_sig = results[results.pval<1e-5]
    logging.info(f'The number of enriched phecode is: {results_sig.shape[0]} (out of {results.shape[0]})')

    logging.info('Reading phecode map...')
    phecode_map = pd.read_csv(args.phecode_map_file, dtype={'Phecode':str})
    phecode_map = phecode_map[['Phecode', 'PhecodeString']].drop_duplicates(ignore_index=True)
    phecode_map.Phecode = phecode_map.Phecode.apply(lambda x: x.strip())
    phecode_map.index = phecode_map.Phecode
    phecode_map.drop(columns=['Phecode'], inplace=True)
    phecode_map = phecode_map.to_dict()
    phecode_map = phecode_map['PhecodeString']

    def find_phecode_string(x):
        x = str(x)
        try:
            s = phecode_map[x]
        except:
            s = 'NA'
        return s

    results_sig.loc[:, 'PhecodeString'] = results_sig.phecode.apply(find_phecode_string)
    results_sig = results_sig.sort_values(by='phecode', ignore_index=True)
    
    # Filter by minimum frequency
    min_phecode_frequency = config.get('min_phecode_frequency', 0.02)
    
    # Try to find case control file to get total number of cases
    case_file_train = output_path / f'case_control_pairs_{prefix}_train.txt'
    case_file_all = output_path / f'case_control_pairs_{prefix}.txt'
    
    total_cases = 0
    if case_file_train.exists():
        logging.info(f'Reading case control file: {case_file_train}')
        case_df = pd.read_csv(case_file_train, sep='\t')
        total_cases = len(case_df['case'].dropna().unique())
    elif case_file_all.exists():
        logging.info(f'Reading case control file: {case_file_all}')
        case_df = pd.read_csv(case_file_all, sep='\t')
        total_cases = len(case_df['case'].dropna().unique())
    else:
        logging.warning("Could not find case control file (checked _train.txt and .txt). Cannot filter by frequency.")
    
    if total_cases > 0:
        logging.info(f'Total number of cases: {total_cases}')
        logging.info(f'Minimum frequency threshold: {min_phecode_frequency} (Count > {total_cases * min_phecode_frequency})')
        n_before = len(results_sig)
        results_sig = results_sig[results_sig['case_count'] > total_cases * min_phecode_frequency]
        n_after = len(results_sig)
        logging.info(f'Filtered {n_before - n_after} phecodes with low frequency.')
    # results_sig.head()

    enriched_phecode = pd.DataFrame(columns=['Phecode', 'Description', 'Count', 'p.value',
                                         'p01', 'p05', 'p10', 'p50', 'p90',
                                         'p95', 'p99', 'max', 'case_to_control_ratio'])

    logging.info('Generating statistics for enrichment analysis...')
    for i in tqdm(range(len(results_sig))):
        control_count = pd.to_numeric(results_sig.iloc[i, 3:-1], errors='coerce').to_list()
        row = results_sig.iloc[i]
        case_count = row['case_count']
        code, pval, desc = row['phecode'], row['pval'], row['PhecodeString']
        
        percentiles = [1, 5, 10, 50, 90, 95, 99]
        stats = [int(np.percentile(control_count, p)) for p in percentiles]
        max_count = int(max(control_count)) if len(control_count) > 0 else 0
        ratio = round(int(case_count) / max_count, 2) if max_count > 0 else 1000

        enriched_phecode.loc[i] = [code, desc, int(case_count), pval] + stats + [max_count, ratio]

    # enriched_phecode = enriched_phecode[enriched_phecode.Description!='NA']
    enriched_phecode.sort_values(by='case_to_control_ratio', ascending=False, inplace=True)
    enriched_phecode.to_csv(output_path / f'{TRAIT}_{prefix}_enriched_phecode.csv', sep='\t', index=False)

    logging.info('\nDone. Report generated.\n\n')

if __name__ == '__main__':
    main()