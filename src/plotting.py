import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional, Union, Dict
import logging


def plot_CM(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    model_type: str,
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
    class_names = ['Control', trait]
    disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X,
            y,
            display_labels=class_names,
            cmap=plt.cm.Blues,
        )
    p = precision_score(y, model.predict(X))
    disp.ax_.set_title(f'Confusion Matrix of {trait} Prediction Model (Precision: {p:.2f}')
    plt.savefig(output_path / f'{prefix}_{model_type}_CM.png', bbox_inches='tight')
    return p

def plot_ROC(
    final_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    trait: str,
    model_type: str,
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
        model_type (str): Model type
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
    plt.plot(fpr, tpr, label=f'{model_type} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {trait} prediction model')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(output_path / f'{prefix}_{model_type}_ROC_curve.png', bbox_inches='tight')
    return auc

def plot_precision_recall(
    final_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    trait: str,
    model_type: str,
    prefix: str
) -> float:
    """
    Plot Precision-Recall curve for testing data.

    Args:
        final_model (Any): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        output_path (Path): Output directory
        trait (str): Trait name
        model_type (str): Model type
        prefix (str): Output file prefix

    Returns:
        float: Average precision score
    """
    y_pred_prob = final_model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)

    plt.figure()
    plt.step(recall, precision, where='post', label=f'{model_type} (AP = {avg_precision:.2f})')
    plt.fill_between(recall, precision, alpha=0.1, step='post')

    positive_ratio = y_test.mean()
    plt.axhline(positive_ratio, color='k', linestyle='--', label=f'Baseline (Prevalence = {positive_ratio:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {trait} prediction model')
    plt.legend(loc='lower left')
    plt.savefig(output_path / f'{prefix}_{model_type}_PR_curve.png', bbox_inches='tight')
    plt.close()

    return avg_precision

def plot_feature_importances(
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

def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    prefix: str,
    model_type: str,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Compute permutation feature importance for any model type.
    Args:
        model (Any): Trained model
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        output_path (Path): Output directory
        prefix (str): Output file prefix
        model_type (str): Model type for naming
        n_repeats (int): Number of permutation repeats
        random_state (int): Random seed
        n_jobs (int): Number of parallel jobs
    Returns:
        pd.DataFrame: Permutation importance results
    """
    logging.info(f'Computing permutation importance for {model_type} model...')
    
    # Compute permutation importance
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring='accuracy'
    )
    
    # Create results DataFrame
    results = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'importance_values': [imp.tolist() for imp in perm_importance.importances]
    })
    
    # Sort by importance
    results = results.sort_values('importance_mean', ascending=False)
    
    # Save results
    results_file = output_path / f'{prefix}_{model_type}_permutation_importance.csv'
    results.to_csv(results_file, index=False)
    logging.info(f'Saved permutation importance results to {results_file}')
    
    return results

def plot_permutation_importance(
    perm_results: pd.DataFrame,
    output_path: Path,
    prefix: str,
    model_type: str,
    n_top: int = 15,
    phecode_map: Optional[Union[Dict[str, str], pd.DataFrame]] = None
) -> None:
    """
    Plot permutation feature importance results.
    Args:
        perm_results (pd.DataFrame): Results from compute_permutation_importance
        output_path (Path): Output directory
        prefix (str): Output file prefix
        model_type (str): Model type for naming
        n_top (int): Number of top features to plot
        phecode_map (Optional[Union[Dict[str, str], pd.DataFrame]]): Mapping from phecode to description
    Returns:
        None
    """
    # Get top features
    top_results = perm_results.head(n_top)
    
    # Map phecodes to string names if mapping is provided
    if phecode_map is not None:
        if isinstance(phecode_map, dict):
            feature_descs = [phecode_map.get(str(f), "") for f in top_results['feature']]
        elif hasattr(phecode_map, "set_index"):
            map_dict = dict(zip(phecode_map['Phecode'].astype(str), phecode_map['Description']))
            feature_descs = [map_dict.get(str(f), "") for f in top_results['feature']]
        else:
            feature_descs = ["" for f in top_results['feature']]
    else:
        feature_descs = ["" for f in top_results['feature']]
    
    # Build ytick labels as "phecode: description"
    feature_labels = [
        f"{str(phecode)}: {desc}" if desc else str(phecode)
        for phecode, desc in zip(top_results['feature'], feature_descs)
    ]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot bars with error bars
    y_pos = np.arange(len(top_results))
    plt.barh(y_pos, top_results['importance_mean'], 
             xerr=top_results['importance_std'], 
             capsize=3, align='center')
    
    # Customize the plot
    plt.yticks(y_pos, feature_labels)
    plt.xlabel('Permutation Importance (Mean ± Std)')
    plt.title(f'Top {n_top} Permutation Feature Importances - {model_type}')
    plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at top
    plt.tight_layout()
    
    # Save the plot
    out_fn = output_path / f'{prefix}_{model_type}_permutation_importance_top{n_top}.png'
    plt.savefig(out_fn, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f'Saved permutation importance plot to {out_fn}')

def interpret_model(
    model: object, 
    data: pd.DataFrame, 
    phecode_map: dict,
    grid: str = None,
    output_path: Path = Path('.'), 
    prefix: str = 'output', 
    plot: str = 'waterfall', 
    show: bool = False, 
    model_type: str = 'tree'
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
        if grid is None: return None
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
    if model_type=='tree':
        print('Using tree explainer')
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    shap_values = explainer(X_phecode)

    idx = find_index_of_grid(data, grid)
    if idx is None and plot == 'waterfall':
        raise ValueError(f"Grid ID {grid} not found in data.")

    # Generate the requested SHAP plot
    if plot == 'waterfall':
        # For binary classification, use the SHAP values for class 1
        shap.plots.waterfall(shap_values[idx, :, 1], show=show)
    elif plot == 'heatmap':
        shap.plots.heatmap(shap_values[:, :, 1], show=show)
    elif plot == 'beeswarm':
        shap.plots.beeswarm(shap_values[:, :, 1], show=show)
    else:
        raise ValueError(f"Unsupported plot type: {plot}")

    # Save the plot if not displaying interactively
    if not show:
        if plot == 'waterfall':
            plt.title(f'{plot.capitalize()} plot for {grid}')
            out_file = output_path / f'{plot}_for_{grid}_{prefix}.png'
        else:
            plt.title(f'{plot.capitalize()} plot')
            out_file = output_path / f'{plot}_{prefix}.png'
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved {plot} plot for {grid if grid else 'all'} to {out_file}")