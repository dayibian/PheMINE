import pandas as pd
import numpy as np
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
import shap
try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is not installed. Please install xgboost to use XG model.")

import matplotlib.pyplot as plt

import joblib
import gzip
from tqdm import tqdm
import yaml

import logging, sys, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True

try:
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Try to open config.yaml in the current directory
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

full_name = {
    'als': 'ALS',
    'ftld': 'FTLD',
    'vasc_dementia': 'Vascular Dementia',
    'lewy_body': 'Lewy Body Dementia',
    'hpp': 'Hypophosphatasia'
}

def setup_log(fn_log: Union[str, Path], mode: str = 'w') -> None:
    '''
    Print log message to console and write to a log file.
    Will overwrite existing log file by default
    Params:
    - fn_log: name of the log file
    - mode: writing mode. Change mode='a' for appending
    '''
    # Remove any existing handlers to avoid duplicate logs
    logging.root.handlers = [] # Remove potential handler set up by others (especially in google colab)
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(filename=fn_log, mode=mode),
                                  logging.StreamHandler()], format='%(message)s')

def process_args() -> argparse.Namespace:
    '''
    Process command-line arguments and set up logging.
    Returns:
        argparse.Namespace: Parsed arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='High-level folder that contains data for each trait', type=str,
                        default='/data100t1/home/biand/Projects/Comorbidity_analysis/data/')
    parser.add_argument('--output_folder', help='High-level folder that contains output for each trait', type=str,
                        default='/data100t1/home/biand/Projects/Comorbidity_analysis/output/')
    parser.add_argument('--trait', help='Trait of interest', type=str, default='als')
    parser.add_argument('--output_prefix', type=str, default='output')
    parser.add_argument('--model_type', type=str, default='CART')
    parser.add_argument('--matched_controls_for_ML', type=int, default=1)
    
    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(args.output_folder) / f'_PheML_{args.trait}.log'
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

def get_phecode_features(
    data_path: Path,
    output_path: Path,
    trait: str,
    prefix: str,
    number_of_cases: int
) -> List[str]:
    '''
    Get enriched phecodes, drop those used for phenotyping.
    Args:
        data_path (Path): Path to data directory
        output_path (Path): Path to output directory
        trait (str): Trait of interest
    Returns:
        List[str]: List of phecode features (excluding those used for phenotyping)
    '''
    
    icd_codes = {
        'als': ['G12.21', 'G12.20', 'G12.24', 'G12.29', '335.20', '335.21', '335.29'],
        'ftld': ['G31.01', 'G31.09', 'G21.1', 'G31.85', '331.11', '331.19', '331.6'],
        'vasc_dementia': ['F01.50', 'F01.51', 'F01.511', 'F01.518', '290.40', '290.41'],
        'lewy_body': ['G31.83', 'G20', 'F02.80', '331.82', '332.0', '294.10'],
        'hpp': ['275.3', 'E83.39']
    }


    phecodes = {
        'hpp': []
    }

    if trait in phecodes:
        excluded_code = phecodes[trait]
    else:
        with gzip.open(data_path / f'{trait}/{trait}_codes_and_dates.csv.gz') as f:
            case_codes = pd.read_csv(f, dtype={'phecode':str})
        case_codes_ = case_codes[case_codes.concept_code.isin(icd_codes[trait])]
        excluded_code = case_codes_.phecode.unique()
    
    # Get enriched phecode
    enrich_results = pd.read_csv(output_path / f'{trait}_{prefix}_enriched_phecode.csv', sep='\t', dtype={'Phecode':str})
    enrich_results = enrich_results[enrich_results.Count > number_of_cases * .01] # Remove those phecodes that has counts less than 1% of case number, regardless of significance
    phecode_features = enrich_results.Phecode.astype(str).unique().tolist()
    
    excluded_set = set(excluded_code)
    phecode_features_ = [code for code in phecode_features if code not in excluded_set] if excluded_set else phecode_features
    return phecode_features_


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'RF',
    random_state: int = 42,
    verbose: int = 2,
    n_jobs: int = -1
) -> Any:
    '''
    Train a machine learning model with hyperparameter tuning.
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_type (str): 'CART', 'RF', 'XG', or 'NN'/'MLP'
        random_state (int): Random seed
        verbose (int): Verbosity level
        n_jobs (int): Number of parallel jobs
    Returns:
        Any: The best trained model
    '''
    if model_type.upper() == 'CART':
        m = X_train.shape[1]
        param_dist = {
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': randint(max(1, m // 2), m)
        }
        base_model = DecisionTreeClassifier(random_state=random_state)
        n_iter = 10
    elif model_type.upper() == 'RF':
        param_dist = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        base_model = RandomForestClassifier(random_state=random_state)
        n_iter = 50
    elif model_type.upper() == 'XG':
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'subsample': [0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        base_model = XGBClassifier(eval_metric='logloss', random_state=random_state, use_label_encoder=False)
        n_iter = 20
    elif model_type.upper() in ['NN', 'MLP']:
        param_dist = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
        base_model = MLPClassifier(max_iter=500, random_state=random_state)
        n_iter = 20
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'CART', 'RF', 'XG', or 'NN'/'MLP'.")

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    return best_model


def plot_CM(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    trait: str,
    prefix: str
) -> float:
    '''
    Plot confusion matrix for testing data.
    Args:
        model (Any): Trained model
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        output_path (Path): Output directory
        trait (str): Trait name
        prefix (str): Output file prefix
    Returns:
        float: Precision score
    '''
    class_names = ['Control', full_name[trait]]
    disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X,
            y,
            display_labels=class_names,
            cmap=plt.cm.Blues,
        )
    p = precision_score(y, model.predict(X))
    disp.ax_.set_title(f'Confusion Matrix of {full_name[trait]} Prediction Model (Precision: {p:.2f}')
    plt.savefig(output_path / f'{trait}_CM_{prefix}.png', bbox_inches='tight')
    return p

def plot_ROC(
    final_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    trait: str,
    prefix: str
) -> float:
    '''
    Plot ROC curve for testing data.
    Args:
        final_model (Any): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        output_path (Path): Output directory
        trait (str): Trait name
        prefix (str): Output file prefix
    Returns:
        float: AUC score
    '''
    y_pred_prob = final_model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Calculate the AUC (Area Under the Curve)
    auc = roc_auc_score(y_test, y_pred_prob)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {full_name[trait]} prediction model')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(output_path / f'{trait}_ROC_curve_{prefix}.png', bbox_inches='tight')
    return auc

def plot_top_feature_importances(
    model: Any,
    X: pd.DataFrame,
    output_path: Path,
    prefix: str,
    n_top: int = 10,
    phecode_map: Optional[Union[Dict[str, str], pd.DataFrame]] = None
) -> None:
    """
    Plot and save the top n feature importances for a fitted RandomForest model.
    Optionally, use phecode_map (dict or DataFrame) to map phecodes to string names for axis labels.
    The y-axis will show "phecode: description" (phecode left, description right).
    Args:
        model (Any): Trained model with feature_importances_
        X (pd.DataFrame): Training features
        output_path (Path): Output directory
        prefix (str): Output file prefix
        n_top (int): Number of top features to plot
        phecode_map (Optional[Union[Dict[str, str], pd.DataFrame]]): Mapping from phecode to description
    Returns:
        None
    """
    if not hasattr(model, "feature_importances_"):
        logging.warning("Model does not have feature_importances_ attribute.")
        return
    importances = model.feature_importances_
    feature_names = X.columns
    # Get indices of top n features
    top_idx = importances.argsort()[::-1][:n_top]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = importances[top_idx]

    # Map phecodes to string names if mapping is provided
    if phecode_map is not None:
        # If dict, use directly; if DataFrame, build dict from columns
        if isinstance(phecode_map, dict):
            feature_descs = [phecode_map.get(str(f), "") for f in top_features]
        elif hasattr(phecode_map, "set_index"):
            # Assume DataFrame with columns 'Phecode' and 'Description'
            map_dict = dict(zip(phecode_map['Phecode'].astype(str), phecode_map['Description']))
            feature_descs = [map_dict.get(str(f), "") for f in top_features]
        else:
            feature_descs = ["" for f in top_features]
    else:
        feature_descs = ["" for f in top_features]

    # Build ytick labels as "phecode: description"
    feature_labels = [
        f"{str(phecode)}: {desc}" if desc else str(phecode)
        for phecode, desc in zip(top_features, feature_descs)
    ]

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(top_features)), top_importances[::-1], align='center')
    plt.yticks(
        range(len(top_features)),
        [feature_labels[i] for i in range(len(top_features)-1, -1, -1)]
    )
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n_top} Feature Importances')
    plt.tight_layout()
    out_fn = output_path / f'{prefix}_rf_feature_importance_top{n_top}.png'
    plt.savefig(out_fn)
    plt.close()
    logging.info(f'Saved top {n_top} feature importance plot to {out_fn}')

def get_cases_and_controls(
    pair_file: Union[str, Path],
    potential_controls: list,
    n_controls_per_case: int = 5,
    use_matched_controls: bool = True
) -> Tuple[List[Any], List[Any]]:
    """
    Reads a case-control pair file and returns lists of case and control IDs.
    Args:
        pair_file (str or Path): Path to the case-control pairs file.
        potential_controls (list): List of unmatched control IDs. Required if use_matched_controls is False.
        n_controls_per_case (int): Number of controls to use per case (max is the number of control columns in the file or number to sample from potential_controls).
        use_matched_controls (bool): Whether to use the controls from the matched control set. If False, use potential_controls.
    Returns:
        cases (list): List of case IDs.
        controls (list): List of unique control IDs (from matched controls or randomly sampled from potential_controls).
    """
    import random

    df = pd.read_csv(pair_file, sep='\t')
    cases = df['case'].dropna().tolist()
    if use_matched_controls:
        # Get only the first n_controls_per_case control columns
        control_cols = [col for col in df.columns if col.startswith('Control')][:n_controls_per_case]
        controls = pd.unique(df[control_cols].values.ravel('K'))
        controls = [c for c in controls if pd.notnull(c)]
    else:
        if potential_controls is None or not isinstance(potential_controls, list) or len(potential_controls) == 0:
            raise ValueError("unmatched_controls must be provided as a non-empty list when use_matched_controls is False.")
        # Sample n_controls_per_case * number of cases, or the max available if not enough
        unmatched_controls = list(set(potential_controls) - set(cases))
        n_controls_total = min(n_controls_per_case * len(cases), len(unmatched_controls))
        controls = random.sample(unmatched_controls, n_controls_total)
    return cases, controls

def interpret_model(
    model: object, 
    data: pd.DataFrame, 
    grid: str, 
    phecode_map: dict, 
    output_path: Path, 
    prefix: str, 
    plot: str = 'waterfall', 
    show: bool = False
) -> None:
    """
    Interpret a trained model for a specific patient (grid) using SHAP values.
    Generates and saves a SHAP plot (waterfall or heatmap) for the given patient.

    Args:
        model: Trained ML model (must be compatible with SHAP).
        data: DataFrame containing patient data (including 'grid' column).
        grid: Patient grid ID to interpret.
        phecode_map: Dictionary mapping phecode to description.
        output_path: Path to save the plot.
        prefix: Prefix for output filename.
        plot: Type of SHAP plot ('waterfall' or 'heatmap').
        show: If True, display the plot; if False, save to file.
    """

    # Helper: Find the row index for the given grid ID
    def find_index_of_grid(data, grid):
        matches = data.index[data['grid'] == grid]
        return matches[0] if not matches.empty else None

    # Extract feature columns (assumes first col is 'grid', last is 'label')
    X_phecode = data.iloc[:, 1:-1].copy()

    # Add descriptions to feature columns using phecode_map (if available)
    X_phecode.columns = [
        f"{col} ({phecode_map[col]})" if col in phecode_map else col
        for col in X_phecode.columns
    ]

    # Use SHAP to explain the model's predictions
    explainer = shap.Explainer(model)
    shap_values = explainer(X_phecode)

    idx = find_index_of_grid(data, grid)
    if idx is None:
        raise ValueError(f"Grid ID {grid} not found in data.")

    # Generate the requested SHAP plot
    if plot == 'waterfall':
        # For binary classification, use the SHAP values for class 1
        shap.plots.waterfall(shap_values[idx, :, 1], show=show)
    elif plot == 'heatmap':
        shap.plots.heatmap(shap_values[:, :, 1], show=show)
    else:
        raise ValueError(f"Unsupported plot type: {plot}")

    # Save the plot if not displaying interactively
    if not show:
        plt.title(f'{plot.capitalize()} plot for {grid}')
        out_file = output_path / f'{plot}_for_{grid}_{prefix}.png'
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved {plot} plot for {grid} to {out_file}")

def main() -> None:
    '''
    Main function to orchestrate data preparation, model training, evaluation, and saving.
    '''
    args = process_args()
    trait = args.trait
    data_path = Path(args.data_folder)
    output_path = Path(args.output_folder)
    prefix = args.output_prefix
    model_type = args.model_type
    use_matched_controls = args.matched_controls_for_ML

    # Import case control and corresponding phecodes
    logging.info('Preparing data for model development...')

    sd_phecode = pd.read_feather(config['phecode_binary_feather_file'])
    all_sd_grids = sd_phecode.grid.to_list()
    case_grid, control_grid = get_cases_and_controls(output_path / f'case_control_pairs_{prefix}.txt', 
                                                     potential_controls=all_sd_grids, 
                                                     use_matched_controls=use_matched_controls
                                                     )
    number_of_cases = len(case_grid)

    # case_grid = list(set(cases.GRID))
    # control_grid = list(controls_clean.sample(n=len(case_grid)*30, random_state=2024).GRID)

    # Generate dataframe for case and control, add labels, and merge them
    case_df = sd_phecode[sd_phecode.grid.isin(case_grid)]
    case_df['label'] = 1
    control_df = sd_phecode[sd_phecode.grid.isin(control_grid)]
    control_df['label'] = 0
    data = pd.concat([case_df, control_df], ignore_index=True)
    # Ensure all feature columns are strings
    # feature_cols = [col for col in data.columns if col not in ['grid', 'label']]
    # data[feature_cols] = data[feature_cols].astype(str)
    # print(data.head())

    phecode_features_ = get_phecode_features(data_path, output_path, trait, prefix, number_of_cases)
    data[['grid']+phecode_features_+['label']].to_csv(data_path / f'{prefix}_data_for_ML.csv', index=False)
    # print(phecode_features_)
    X_train, X_test, y_train, y_test = train_test_split(data[phecode_features_], data.label, train_size=0.8,
                                                        random_state=2024, stratify=data.label)

    logging.info('Training the model...')
    final_model = train_model(X_train, y_train, model_type=model_type)

    logging.info('Reading phecode map...')
    phecode_map = pd.read_csv(config['phecode_map_file'], dtype={'Phecode':str})
    phecode_map = phecode_map[['Phecode', 'PhecodeString']].drop_duplicates(ignore_index=True)
    phecode_map.Phecode = phecode_map.Phecode.apply(lambda x: x.strip())
    phecode_map.index = phecode_map.Phecode
    phecode_map.drop(columns=['Phecode'], inplace=True)
    phecode_map = phecode_map.to_dict()
    phecode_map = phecode_map['PhecodeString']

    logging.info('Plotting model results...')
    # Call the feature importance plotting function
    plot_top_feature_importances(final_model, X_train, output_path, prefix, n_top=10, phecode_map=phecode_map)
    precision = plot_CM(final_model, X_test, y_test, output_path, trait, prefix)
    auc = plot_ROC(final_model, X_test, y_test, output_path, trait, prefix)
    logging.info(f'Precision is: {precision:.2f}')
    logging.info(f'AUC is: {auc:.2f}')

    logging.info('Saving model...')
    joblib.dump(final_model, output_path / f'PheML_{model_type}_{prefix}.model')
    logging.info('Done. Model building completed.')

if __name__ == '__main__':
    main()
    #TODO: Add explainability to the model