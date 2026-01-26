from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from collections import defaultdict
import re
import editdistance
import string
import unicodedata
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams


LIGATURES = {
    'ﬀ': 'ff',
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬃ': 'ffi',
    'ﬄ': 'ffl',
    'ﬅ': 'ft',
    'ﬆ': 'st',
    'Æ': 'AE',
    'æ': 'ae',
    'Œ': 'OE',
    'œ': 'oe',
    'ᵫ': 'ue',
    'ꜳ': 'aa',
    'Ꜵ': 'ao',
    'ꜵ': 'au',
    'Ꜷ': 'av',
    'ꜷ': 'av',
    'Ꜹ': 'ay',
    'ꜹ': 'et',
    'Ꜻ': 'is',
    'ꜻ': 'us',
    'Ꜽ': 'vy',
    'Ĳ': 'IJ',
    'ĳ': 'ij',
    'Ꝏ': 'oo',
    'ꝏ': 'oo',
    '&': 'et',
}

LIGATURE_PATTERN = re.compile("|".join(map(re.escape, LIGATURES.keys())))


def edit_thres_match(target: str, prediction: str, theta: float = 0.3):
    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), 1)
    return 1.0 if normalized_ld <= theta else 0.0


def metric_calculate(
    targets: Sequence[Sequence[Any]],
    predictions: Sequence[Any],
    metric_fn: Callable[[Any, Any], Any],
    normalize_fn: Callable[[Any], Any] = lambda v: v):
    """Aggregate target-prediction pair metrics over a dataset."""
    assert len(targets) == len(predictions)
    total = 0
    scores = []
    for prediction, target in zip(predictions, targets):
        p = normalize_fn(prediction)
        score = max(metric_fn(normalize_fn(t), p) for t in target)
        scores.append(score)
        total += score
    score = (100.0 * total) / len(targets)
    return score, scores


def extract_ngrams_from_list(text_list, n):
    """
    Extract in-sentence n-grams from a list of strings.
    
    Preprocessing includes:
    - Replace ligatures
    - Lowercasing
    - Removing digits
    - Merging hyphenated words
    - Removing all punctuation
    - Sentence splitting
    - Tokenization per sentence
    - No n-grams across sentence boundaries

    Args:
    text_list (list of str): The input list of strings.
    n (int): The length of the n-grams.

    Returns:
    list of str: A list of extracted n-grams.
    """
    all_ngrams = []

    for text in text_list:
        # Sentence segmentation
        text = text.replace("\\n", "\n")
        paragraphs = text.split("\n\n")
        sentences = []
        for paragraph in paragraphs:
            sentences.extend(sent_tokenize(paragraph))
        for sentence in sentences:
            # Normalize Unicode first
            sentence = unicodedata.normalize("NFKD", sentence)
            # Replace ligatures (in case some survived normalization)
            sentence = LIGATURE_PATTERN.sub(lambda m: LIGATURES[m.group(0)], sentence)
            # Lowercase
            sentence = sentence.lower()
            # Remove digits
            sentence = re.sub(r'\d+', '', sentence)
            # Merge hyphenated words
            sentence = re.sub(r'(\w)\s*-\s*(\w)', r'\1\2', sentence)
            # Remove punctuation
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            # Tokenize into words
            tokens = word_tokenize(sentence)
            # Extract n-grams
            n_grams = [" ".join(gram) for gram in ngrams(tokens, n)]
            all_ngrams.extend(n_grams)

    return all_ngrams


def metric_calculate_f1(gts, preds, theta=0.3, n_gram=-1, micro=False):
    prefix = f'n{n_gram}_' if n_gram > 0 else ''
    if any(isinstance(gt, str) for gt in gts): prefix = 's_' + prefix

    # Initialize dictionaries to store final scores and per-case scores
    metric2score = {}
    metric2scores = {
        f'{prefix}precision': [],
        f'{prefix}recall': [],
        f'{prefix}f1': [],
    }
    if micro:
        metric2scores[f'{prefix}tp'] = []
        metric2scores[f'{prefix}fp'] = []
        metric2scores[f'{prefix}fn'] = []

    total_tp = total_fp = total_fn = 0  # For micro calculation
    num_cases = len(gts)  # Number of cases to evaluate

    # Iterate through each target and prediction pair
    for target_list, pred_list in zip(gts, preds):
        tp = 0  # True Positives for this case

        # For each prediction in pred_list, check if it matches any target in target_list
        matched_targets = set()
        matched_preds = set()

        # TODO: Temporarily treat empty GT as 0 score to omit its impact on average, should be separate metrics
        if target_list == "" or not target_list: 
            metric2scores[f'{prefix}precision'].append(0)
            metric2scores[f'{prefix}recall'].append(0)
            metric2scores[f'{prefix}f1'].append(0)
            num_cases -= 1
            continue

        if isinstance(target_list, str): target_list = [target_list] 

        if n_gram > 0:
            target_list = extract_ngrams_from_list(target_list, n_gram)
            pred_list = extract_ngrams_from_list(pred_list, n_gram)

        for i, pred in enumerate(pred_list):
            for j, target in enumerate(target_list):
                if j in matched_targets:
                    continue
                if edit_thres_match(target, pred, theta) == 1.0:
                    tp += 1
                    matched_targets.add(j)
                    matched_preds.add(i)
                    break

        fp = len(pred_list) - len(matched_preds)
        fn = len(target_list) - len(matched_targets)

        # Calculate precision, recall, and F1 for this case
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero
        if not target_list and not pred_list:
            precision = recall = 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F1 score

        # Store the precision, recall, and F1 for this case in the corresponding lists
        metric2scores[f'{prefix}precision'].append(precision * 100)
        metric2scores[f'{prefix}recall'].append(recall * 100)
        metric2scores[f'{prefix}f1'].append(f1 * 100)

        # For micro calculation, accumulate TP, FP, FN
        if micro:
            total_tp += tp
            total_fp += fp
            total_fn += fn
            # Store the tp, fp, fn for this case in the corresponding lists
            metric2scores[f'{prefix}tp'].append(tp)
            metric2scores[f'{prefix}fp'].append(fp)
            metric2scores[f'{prefix}fn'].append(fn)

    # Compute macro average precision, recall, and F1
    macro_precision = sum(metric2scores[f'{prefix}precision']) / num_cases if num_cases > 0 else 0
    macro_recall = sum(metric2scores[f'{prefix}recall']) / num_cases if num_cases > 0 else 0
    macro_f1 = sum(metric2scores[f'{prefix}f1']) / num_cases if num_cases > 0 else 0

    # Store the macro average scores in the metric2score dictionary
    metric2score[f'{prefix}precision'] = round(macro_precision, 2)
    metric2score[f'{prefix}recall'] = round(macro_recall, 2)
    metric2score[f'{prefix}f1'] = round(macro_f1, 2)

    # If micro is True, calculate micro average precision, recall, and F1
    if micro:
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        metric2score[f'micro_{prefix}precision'] = round(micro_precision * 100, 2)
        metric2score[f'micro_{prefix}recall'] = round(micro_recall * 100, 2)
        metric2score[f'micro_{prefix}f1'] = round(micro_f1 * 100, 2)

    return metric2score, metric2scores


def metric_calculate_recall_per_category_asym(gts, preds, theta=0.3, n_gram=-1):
    """
    Calculates recall for each category based on categorized ground truths (gts)
    and uncategorized predictions (preds).

    This is an "asymmetric" calculation because the structure of gts and preds is different.
    """
    # 'total_gt' represents the sum of TP and FN for that category.
    category_stats = defaultdict(lambda: {"tp": 0, "total_gt": 0})
    per_category_page_scores = defaultdict(list)

    # Pre-scan to find all unique category labels for consistent reporting
    all_categories = set()
    for gt_dict in gts:
        all_categories.update(gt_dict.keys())
    all_categories = sorted(list(all_categories))

    # Iterate through each example pair (gt_dict, pred_list)
    for gt_dict, pred_list_original in zip(gts, preds):

        page_stats = defaultdict(lambda: {"tp": 0, "total_gt": 0})
        pred_list = list(pred_list_original)

        # First, calculate the stats for the current page
        for category, target_list_original in gt_dict.items():
            target_list = list(target_list_original)

            if n_gram > 0:
                target_list = extract_ngrams_from_list(target_list, n_gram)
                pred_list = extract_ngrams_from_list(pred_list, n_gram)
            
            page_stats[category]["total_gt"] += len(target_list)

            matched_pred_indices = set()
            for target_text in target_list:
                for i, pred_text in enumerate(pred_list):
                    if i in matched_pred_indices:
                        continue
                    
                    if edit_thres_match(target_text, pred_text, theta) == 1.0:
                        page_stats[category]["tp"] += 1
                        matched_pred_indices.add(i)
                        break
            
            # Add this page's raw stats to the overall aggregator
            category_stats[category]["tp"] += page_stats[category]["tp"]
            category_stats[category]["total_gt"] += page_stats[category]["total_gt"]

        # Now, iterate through all possible categories to build the per-page lists
        for category in all_categories:
            if category in gt_dict:
                stats = page_stats[category]
                tp = stats["tp"]
                total_gt = stats["total_gt"]
                recall = (tp / total_gt) * 100.0 if total_gt > 0 else 0.0
                per_category_page_scores[category].append(recall)
            else:
                # Append the placeholder for categories not present on this page
                per_category_page_scores[category].append(None)

    # New logic: Calculate the average of per-page scores for each category.
    
    macro_average_scores = {}
    for category, page_scores in per_category_page_scores.items():
        # Filter out the 'None' placeholders for pages where the category didn't appear
        valid_scores = [score for score in page_scores if score is not None]
        
        # Calculate the average if there are any valid scores
        if valid_scores:
            average_recall = np.mean(valid_scores)
        else:
            # If a category never appeared in any ground truth, its average recall is 0
            average_recall = 0.0
            
        macro_average_scores[category] = round(average_recall, 2)

    # The two dictionaries are now calculated. Now we add the "_recall" suffix.
    overall_recall_scores = {f"{k}_recall": v for k, v in macro_average_scores.items()}
    per_category_page_scores = {f"{k}_recall": v for k, v in per_category_page_scores.items()}

    return overall_recall_scores, dict(per_category_page_scores)

def metric_calculate_f1_per_category(gts, preds, theta=0.3, n_gram=-1):
    """
    Calculates precision, recall, and F1 for each category based on categorized ground truths (gts)
    and categorized predictions (preds). All items are input as dictionaries mapping category labels to lists of strings.
    """
    # 'total_gt' represents the sum of TP and FN for that category.
    # category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "total_pred": 0})
    per_category_page_scores = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})

    # Pre-scan to find all unique category labels for consistent reporting
    all_categories = set()
    for gt_dict in gts:
        all_categories.update(gt_dict.keys())
    all_categories = sorted(list(all_categories))

    # Iterate through each example pair (gt_dict, pred_dict)
    for gt_dict, pred_dict in zip(gts, preds):

        page_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "total_pred": 0})

        # First, calculate the stats for the current page
        for category in all_categories:
            stats = page_stats[category]
            target_list_original = gt_dict.get(category, [])
            pred_list_original = pred_dict.get(category, [])

            target_list = list(target_list_original)
            pred_list = list(pred_list_original)

            if n_gram > 0:
                target_list = extract_ngrams_from_list(target_list, n_gram)
                pred_list = extract_ngrams_from_list(pred_list, n_gram)
            
            stats["total_gt"] += len(target_list)
            stats["total_pred"] += len(pred_list)

            matched_pred_indices = set()
            matched_target_indices = set()
            for target_idx, target_text in enumerate(target_list):
                for pred_idx, pred_text in enumerate(pred_list):
                    if pred_idx in matched_pred_indices or target_idx in matched_target_indices:
                        continue
                    
                    if edit_thres_match(target_text, pred_text, theta) == 1.0:
                        stats["tp"] += 1
                        matched_pred_indices.add(pred_idx)
                        matched_target_indices.add(target_idx)
                        break

            stats["fn"] += len(target_list) - len(matched_target_indices)
            stats["fp"] += len(pred_list) - len(matched_pred_indices)

            # --- Calculate Precision, Recall, and F1 for this specific page ---
            precision = (stats["tp"] / stats["total_pred"]) * 100.0 if stats["total_pred"] > 0 else 0.0
            recall = (stats["tp"] / stats["total_gt"]) * 100.0 if stats["total_gt"] > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # --- Append scores or a placeholder to the per_category_page_scores lists ---
            if stats["total_gt"] == 0 and stats["total_pred"] == 0:
                # This category was not relevant to this page (not in GT and not in predictions)
                per_category_page_scores[category]["precision"].append(None)
                per_category_page_scores[category]["recall"].append(None)
                per_category_page_scores[category]["f1"].append(None)
            else:
                per_category_page_scores[category]["precision"].append(precision)
                per_category_page_scores[category]["recall"].append(recall)
                per_category_page_scores[category]["f1"].append(f1)

    # After the main loop has processed all pages, calculate the final scores
    macro_average_scores = {}
    final_per_page_scores = {}
    for category, metric_lists in per_category_page_scores.items():
        for metric_name, score_list in metric_lists.items():
            # Filter out 'None' placeholders for pages where the category didn't appear
            valid_scores = [s for s in score_list if s is not None]
            
            # Calculate the average if there are any valid scores
            average_score = np.mean(valid_scores) if valid_scores else 0.0

            # Create the final key, e.g., "Bilingual_cat_recall"
            final_key = f"{category}_cat_{metric_name}"
            macro_average_scores[final_key] = round(average_score, 2)

            # Also update the per-category page scores to have the final key format
            final_per_page_scores[final_key] = score_list
            
    return macro_average_scores, final_per_page_scores


def metric_calculate_case_f1(
    targets: List[Any],
    predictions: List[Any],
    normalize_fn: Callable[[Any], Any] = lambda v: v):
    """
    Calculate precision, recall, and F1 score for positive and negative cases,
    the macro-F1 score, Matthews correlation coefficient (MCC), and per-case classification indicators.

    Args:
        targets: A list of ground truth labels. An empty list/string or None
                 represents a negative case. Anything else is positive.
        predictions: A list of predicted labels. The truthiness after applying
                     normalize_fn determines positive/negative prediction.
        normalize_fn: A function to normalize predictions before evaluation.

    Returns:
        A tuple containing two dictionaries:
        1. metric2score (Dict[str, float]): Aggregated metrics.
           - case_pos_precision: Precision for the positive class.
           - case_pos_recall: Recall for the positive class.
           - case_pos_f1: F1 score for the positive class.
           - case_neg_precision: Precision for the negative class.
           - case_neg_recall: Recall for the negative class.
           - case_neg_f1: F1 score for the negative class.
           - case_macro_f1: Macro-averaged F1 score.
           - case_accuracy: Overall accuracy.
           - case_mcc: Matthews correlation coefficient (scaled to percentage).
        2. metric2scores (Dict[str, List[int]]): Per-case indicators.
           - case_tp_indicators: List of 1s (TP) and 0s for each case.
           - case_fp_indicators: List of 1s (FP) and 0s for each case.
           - case_fn_indicators: List of 1s (FN) and 0s for each case.
           - case_tn_indicators: List of 1s (TN) and 0s for each case.
           - case_correctness_indicators: List of 1s (correct) and 0s (incorrect) for each case.
    """
    assert len(targets) == len(predictions), "Targets and predictions must have the same length."

    tp_total = 0  # Total True Positives
    fp_total = 0  # Total False Positives
    tn_total = 0  # Total True Negatives
    fn_total = 0  # Total False Negatives

    # For metric2scores
    case_tp_indicators = []
    case_fp_indicators = []
    case_fn_indicators = []
    case_tn_indicators = []
    case_correctness_indicators = []

    if not targets: # Handle empty input case
        metric2score = {
            'case_pos_precision': 0.0,
            'case_pos_recall': 0.0,
            'case_pos_f1': 0.0,
            'case_neg_precision': 0.0,
            'case_neg_recall': 0.0,
            'case_neg_f1': 0.0,
            'case_macro_f1': 0.0,
            'case_accuracy': 0.0,
            'case_mcc': 0.0,
        }
        metric2scores = {
            'case_tp_indicators': [],
            'case_fp_indicators': [],
            'case_fn_indicators': [],
            'case_tn_indicators': [],
            'case_correctness_indicators': []
        }
        return metric2score, metric2scores

    for target, prediction in zip(targets, predictions):
        normalized_prediction = normalize_fn(prediction)
        is_positive_target = not (target == None or target == [""] or target == [] or target == "")
        is_positive_prediction = not (normalized_prediction is None or normalized_prediction == [""] or normalized_prediction == [])

        is_tp, is_fp, is_fn, is_tn = 0, 0, 0, 0
        is_correct = 0

        if is_positive_target and is_positive_prediction:
            tp_total += 1
            is_tp = 1
            is_correct = 1
        elif not is_positive_target and is_positive_prediction:
            fp_total += 1
            is_fp = 1
        elif not is_positive_target and not is_positive_prediction:
            tn_total += 1
            is_tn = 1
            is_correct = 1
        elif is_positive_target and not is_positive_prediction:
            fn_total += 1
            is_fn = 1
        
        case_tp_indicators.append(is_tp)
        case_fp_indicators.append(is_fp)
        case_fn_indicators.append(is_fn)
        case_tn_indicators.append(is_tn)
        case_correctness_indicators.append(is_correct)

    # Calculate metrics for the positive class
    pos_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    pos_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0.0

    # Calculate metrics for the negative class
    neg_precision = tn_total / (tn_total + fn_total) if (tn_total + fn_total) > 0 else 0.0
    neg_recall = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0.0
    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0.0

    macro_f1 = (pos_f1 + neg_f1) / 2.0
    accuracy = (tp_total + tn_total) / len(targets) if len(targets) > 0 else 0.0

    # Matthews correlation coefficient (MCC)
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    numerator = (tp_total * tn_total) - (fp_total * fn_total)
    denom_terms = (
        (tp_total + fp_total) *
        (tp_total + fn_total) *
        (tn_total + fp_total) *
        (tn_total + fn_total)
    )
    if denom_terms <= 0:
        mcc = 0.0
    else:
        mcc = numerator / np.sqrt(denom_terms)

    metric2score = {
        'case_pos_precision': round(pos_precision * 100, 2),
        'case_pos_recall': round(pos_recall * 100, 2),
        'case_pos_f1': round(pos_f1 * 100, 2),
        'case_neg_precision': round(neg_precision * 100, 2),
        'case_neg_recall': round(neg_recall * 100, 2),
        'case_neg_f1': round(neg_f1 * 100, 2),
        'case_macro_f1': round(macro_f1 * 100, 2),
        'case_accuracy': round(accuracy * 100, 2),
        'case_mcc': round(mcc * 100, 2),  
    }

    metric2scores = {
        'case_tp_indicators': case_tp_indicators,
        'case_fp_indicators': case_fp_indicators,
        'case_fn_indicators': case_fn_indicators,
        'case_tn_indicators': case_tn_indicators,
        'case_correctness_indicators': case_correctness_indicators
    }

    return metric2score, metric2scores


def calculate_token_length_ratio_stats(
    targets: List[str],
    predictions: List[str]):
    """
    Calculates the average token length ratio and per-case token length ratios.
    The ratio is (num_prediction_tokens / num_target_tokens).
    The aggregated metric (average ratio) is "lower is better". Only for negative cases.

    Args:
        targets: A list of target text strings.
        predictions: A list of predicted text strings.
    """
    assert len(targets) == len(predictions), "Targets and predictions must have the same length."

    if not targets:
        return 0.0, []

    per_case_ratios_percent: List[float] = []

    for target, pred in zip(targets, predictions):
        if isinstance(target, str): target = [target] 

        target_tokens = extract_ngrams_from_list(target, 1)
        pred_tokens = extract_ngrams_from_list(pred, 1)

        num_target_tokens = len(target_tokens)
        num_pred_tokens = len(pred_tokens)

        case_ratio_raw: float
        if num_target_tokens > 0:
            case_ratio_raw = num_pred_tokens / num_target_tokens
        else:  
            if num_pred_tokens == 0:
                case_ratio_raw = 1.0  # Both empty, ratio is 1.0 (pred is 100% of target length)
            else: 
                case_ratio_raw = 0.0  # Should be very rare, nearly impossible, but handle it
        
        per_case_ratios_percent.append(case_ratio_raw * 100.0)

    # Calculate average.
    if not per_case_ratios_percent: # Should not happen if initial check passes and targets is not empty
        overall_avg_token_length_ratio_percent = 0.0
    else:
        # Sum will propagate float('inf') if present
        current_sum = sum(per_case_ratios_percent)
        overall_avg_token_length_ratio_percent = current_sum / len(per_case_ratios_percent)
    
    return overall_avg_token_length_ratio_percent, per_case_ratios_percent


def doc_evaluate(
    metric: str,
    targets: Sequence[Sequence[Any]],
    predictions: Sequence[Any],
    **kwargs):

    assert metric in ['N1F1s', 'CaseF1', 'TokenRatio',
                    'micro_N1F1s', 'micro_N2F1s', 'micro_N3F1s', 'micro_N4F1s',
                    'N1CategorizedRecall', 'N1CategorizedF1']
    
    # micro F1 will be calculated in metric_calculate_f1
    if 'micro' in metric:
        metric = metric.replace('micro_', '')
        need_micro = True
    else:
        need_micro = False

    if metric == 'N1F1s':
        theta = kwargs.get('theta', 0.2)
        score, scores = metric_calculate_f1(targets, predictions, theta=theta, n_gram=1, micro=need_micro)
    elif metric == 'N1CategorizedRecall':
        theta = kwargs.get('theta', 0.2)
        score, scores = metric_calculate_recall_per_category_asym(targets, predictions, theta=theta, n_gram=1)
    elif metric == 'N1CategorizedF1':
        theta = kwargs.get('theta', 0.2)
        score, scores = metric_calculate_f1_per_category(targets, predictions, theta=theta, n_gram=1)
    elif metric == 'CaseF1':
        score, scores = metric_calculate_case_f1(targets, predictions)  
    elif metric == 'TokenRatio':
        score, scores = calculate_token_length_ratio_stats(targets, predictions)
    return score, scores 

