import json
import jsonfinder
import jsonlines
import os
from icecream import ic
import argparse
from pathlib import Path
import re
import ast
from collections import defaultdict
import csv

from evaluator import doc_evaluate


def read_jsonl(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            lines.append(line)
    return lines


def save_jsonl(data, filename, print_log=True):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))
        
    if print_log:
        print('save %d samples to %s' % (len(data), filename))


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_json(data, file_path, indent=4, print_log=True):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

    if print_log:
        print('save %d samples to %s' % (len(data), file_path))


def append_metrics_to_dynamic_csv(
    run_data_dict,
    csv_filepath,
    run_id
):
    """
    Appends a new run's data to a CSV file, using a fixed signature.
    It dynamically adds new columns if metrics from the new run don't exist yet.
    The header for the first column is hardcoded as 'run_id'.

    Args:
        run_data_dict (dict): A dictionary containing ONLY the metric scores.
        csv_filepath (str or Path): The path to the master summary CSV file.
        run_id (str): The unique ID for this run (e.g., a test name). This value
                      will populate the first column for the new row.
    """
    csv_path = Path(csv_filepath)
    run_id_column_name = "run_id" # Hardcode the name for the first column

    # 1. Prepare the new row data as a complete dictionary
    # Combine the run identifier (value) with the metrics data.
    new_row = {run_id_column_name: run_id, **run_data_dict}

    # 2. Read existing data and header from the CSV file (if it exists)
    existing_data = []
    header = [run_id_column_name]  # Default header
    
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    header = reader.fieldnames
                existing_data = list(reader)
        except Exception as e:
            print(f"Warning: Could not read existing file '{csv_path}'. A new one will be created. Error: {e}")
            header = [run_id_column_name]

    # 3. Dynamically update the header with any new keys
    for key in new_row.keys():
        if key not in header:
            header.append(key)

    # 4. Merge or update existing row if run_id exists
    updated = False
    for row in existing_data:
        if row.get(run_id_column_name) == run_id:
            # Update only provided keys, keep others unchanged
            row.update(run_data_dict)
            updated = True
            break

    if not updated:
        existing_data.append(new_row)

    # 5. Rewrite the file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, restval='N/A')
        writer.writeheader()
        writer.writerows(existing_data)


def parser_str(s):
    return s.replace('\\n', ' ').replace('\n', ' ').strip()


def extract_json_dict_from_llm_output(llm_output: str) -> dict | None:
    """
    Robustly extracts a JSON dictionary (object) from a string returned by an LLM.

    This function attempts two strategies:
    1. First, it looks for a JSON object enclosed in Markdown code fences (```json ... ```).
    2. If that fails, it looks for the first '{' and the last '}' to extract
       the outermost object.
    
    It returns the parsed dictionary, or None if no valid JSON object can be found.

    Args:
        llm_output: The raw string output from the language model.

    Returns:
        A Python dictionary if a valid JSON object is found, otherwise None.
    """
    # Strategy 1: Look for JSON within Markdown code fences (e.g., ```json ... ```)
    # The [\s\S]*? part is a non-greedy way to match any character, including newlines.
    match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output)
    if match:
        json_str = match.group(1)
        try:
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, dict):
                return parsed_json
        except json.JSONDecodeError:
            print(f"[PARSE DICT] Found a JSON code block, but it was malformed.\nContent: {json_str}")
            try:
                parsed_json = ast.literal_eval(json_str)
                if isinstance(parsed_json, dict):
                    print(f"[PARSE DICT] Successfully parsed dict using ast.literal_eval in the JSON code block.")
                    return parsed_json
            except Exception as e:
                print(f"[PARSE DICT] ast.literal_eval also failed after finding JSON code block: {e}")
            # Fall through to the next strategy

    # Strategy 2: Use jsonfinder to locate any JSON objects in the string
    try:
        final_dict = None
        for _, _, obj in jsonfinder.jsonfinder(llm_output):
            if isinstance(obj, dict):
                print(f"[PARSE DICT] Successfully found a dict using jsonfinder.")
                final_dict = obj  # Keep updating with the latest found dict
        if final_dict is not None:
            return final_dict
    except Exception as e:
        print(f"[PARSE DICT] jsonfinder also failed: {e}")

    # If all strategies fail, return None
    return None


def compute_category_metrics(gts_raw, metric2scores):
    """
    Compute per-category average metrics and count, based only on raw GT data.

    Args:
        gts_raw (list of dict): Each with 'categories': list of str
        metric2scores (dict): {metric_name: list of scores per instance}

    Returns:
        dict: {metric1: {cat1: avg, cat2: avg}, ..., count: {cat1: n, cat2: n}}
    """
    category_metrics = defaultdict(lambda: defaultdict(list))  # {metric: {category: [scores]}}
    category_counts = defaultdict(int)

    for i, gt_raw in enumerate(gts_raw):
        if 'data' in gt_raw: gt_raw = gt_raw['data']
        categories = gt_raw.get('categories', [])

        if type(categories) == str: categories = [cat_str.strip() for cat_str in categories.split(',')]

        if not categories:
            categories = ['NA']
        else:
            if "Full Latin" not in categories:
                categories.append('Partial Latin')
            categories.append('LATIN')
        for category in categories:
            category_counts[category] += 1
            for metric, scores in metric2scores.items():
                if scores[i] is not None: category_metrics[metric][category].append(scores[i])

    summary = {}
    for metric, cat_scores in category_metrics.items():
        summary[metric] = {cat: round(sum(scores)/len(scores), 2) if scores else 0.0
                           for cat, scores in cat_scores.items()}
        
    # Micro P, R, F1
    for metric in category_metrics:
        if 'tp' in metric and 'case' not in metric:
            prefix = '_'.join(metric.split('_')[:-1])
            summary[f'micro_{prefix}_precision'] = {}
            summary[f'micro_{prefix}_recall'] = {}
            summary[f'micro_{prefix}_f1'] = {}

            avg_tp_in_categories = summary[metric]
            avg_fp_in_categories = summary[f'{prefix}_fp']
            avg_fn_in_categories = summary[f'{prefix}_fn']
            for cat in avg_tp_in_categories:
                if cat == 'NA': continue
                avg_tp = avg_tp_in_categories[cat]
                avg_fp = avg_fp_in_categories[cat]
                avg_fn = avg_fn_in_categories[cat]
                summary[f'micro_{prefix}_precision'][cat] = round(avg_tp / (avg_tp + avg_fp) * 100, 2) if (avg_tp + avg_fp) > 0 else 0.0
                summary[f'micro_{prefix}_recall'][cat] = round(avg_tp / (avg_tp + avg_fn) * 100, 2) if (avg_tp + avg_fn) > 0 else 0.0
                summary[f'micro_{prefix}_f1'][cat] = round(2 * (summary[f'micro_{prefix}_precision'][cat] * summary[f'micro_{prefix}_recall'][cat]) / (summary[f'micro_{prefix}_precision'][cat] + summary[f'micro_{prefix}_recall'][cat]), 2) if (summary[f'micro_{prefix}_precision'][cat] + summary[f'micro_{prefix}_recall'][cat]) > 0 else 0.0
        
            del summary[metric]
            del summary[f'{prefix}_fp']
            del summary[f'{prefix}_fn']

    summary['count'] = dict(category_counts)
    return summary


def llm_text_detection_eval(metric_names=["CaseF1", "TokenRatio", "N1F1s", "N1CategorizedRecall"], 
                            result_path='', 
                            gt_path='', 
                            save_path='', 
                            summary_dir='',
                            save_each_eval=True,
                            save_cat_eval=True,
                            sample_num=-1,
                            **kwargs):
    
    if not Path(result_path).exists():
        ic('not exists',result_path)
        return
    ic(result_path)

    preds_raw = read_json(result_path)
    gts_raw = read_json(gt_path)

    if sample_num <= 0:
        sample_num = len(preds_raw)
        assert (len(gts_raw) >= len(preds_raw))

    preds_raw = preds_raw[:sample_num]
    gts_raw = gts_raw[:sample_num]

    preds = [sample['model_pred'] for sample in preds_raw]
    gts = []

    # Initialize special ground truth lists
    if any("F1s" in s for s in metric_names):
        gts = []

    if any("Categorized" in s for s in metric_names):
        cat_gts = [defaultdict(list) for _ in range(len(gts_raw))]

    if any("CategorizedF1" in s for s in metric_names):
        cat_preds = [defaultdict(list) for _ in range(len(preds_raw))]

    if "TokenRatio" in metric_names:
        page_texts = []


    success_parse_count = 0
    # Collect both ground truth and prediction data structures
    for idx, sample in enumerate(gts_raw):
        # To align two different data formats (should be resolved in data further)
        page_anno = sample.get('data', sample)

        if any("Categorized" in s for s in metric_names):
            for annotation in sample.get('annotations', []):
                for anno_res in annotation.get('result', []):
                    if anno_res.get('from_name') == 'category' and anno_res.get('type') == 'labels':
                        if 'value' in anno_res and 'labels' in anno_res['value'] and anno_res['value']['labels']:
                            category_label = anno_res['value']['labels'][0]
                            text_snippet = anno_res['value'].get('text', '')
                            cat_gts[idx][category_label].append(parser_str(text_snippet))
        
        if any("F1s" in s for s in metric_names):
            if any("CategorizedF1" in s for s in metric_names): # use contatenated categorized GT for categorized predictions in overall score
                string_gt = " ".join([text for texts in cat_gts[idx].values() for text in texts])
                gts.append(parser_str(string_gt))
            else:
                gts.append(parser_str(page_anno['pageLatin'])) # use string GT rather than list GT for cleaned n-gram evaluation

        # Collect predicted data structures
        pred_item = preds[idx]
        if isinstance(pred_item, str):
            preds[idx] = []

            success_flag = True
            # First, try to extract JSON dict from the string
            json_dict = extract_json_dict_from_llm_output(pred_item)
            if json_dict:
                for v in json_dict.values():
                    if isinstance(v, list):
                        if all(isinstance(i, str) for i in v):
                            preds[idx].extend([parser_str(i) for i in v])
                        else:
                            print(f"[PARSE] Dict with value is list but contains non-strings: {v}")
                            success_flag = False
                            preds[idx].extend([parser_str(str(i)) for i in v])
                    elif isinstance(v, str):
                        try:
                            candidate_list = ast.literal_eval(v)
                            if all(isinstance(i, str) for i in candidate_list):
                                preds[idx].extend([parser_str(i) for i in candidate_list])
                            else:
                                print(f"[PARSE] Dict with value is str but parsed to list contains non-strings: {candidate_list}")
                                success_flag = False
                                preds[idx].extend([parser_str(str(i)) for i in candidate_list])
                        except:
                            preds[idx].append(v)
                            print(f"[PARSE] Dict with value is str but not list: {v}")
                            success_flag = False

                if any("CategorizedF1" in s for s in metric_names):
                    cat_preds[idx] = json_dict if all(isinstance(i, list) for i in json_dict.values()) else cat_preds[idx]

            # Otherwise, try to extract list from the string
            else:    
                try:
                    matches = [obj for _,_,obj in jsonfinder.jsonfinder(pred_item)]
                except Exception as e:
                    print(f"[PARSE JSON] jsonfinder failed: {e}")
                    matches = []
                    
                if not matches: 
                    print(f"[PARSE JSON] No valid JSON found in: {pred_item}")
                    success_flag = False

                for candidate in matches:
                    if isinstance(candidate, list): 
                        if all(isinstance(i, str) for i in candidate):
                            preds[idx] = [parser_str(i) for i in candidate]
                        else:
                            print(f"[PARSE] Non-string element in list: {candidate}")
                            success_flag = False
                            preds[idx] = [parser_str(str(i)) for i in candidate]

                    else:
                        try:
                            candidate_list = ast.literal_eval(candidate)
                            if all(isinstance(i, str) for i in candidate_list):
                                preds[idx] = [parser_str(i) for i in candidate_list]
                            else:
                                print(f"[PARSE] Parsed candidate but contains non-strings: {candidate_list}")
                                success_flag = False
                                preds[idx] = [parser_str(str(i)) for i in candidate_list]
                        except Exception as e:
                            if isinstance(candidate, str):
                                preds[idx].append(parser_str(candidate))
                                print(f"[PARSE] Candidate parse failed: {candidate} | Error: {e}")
                                success_flag = False

            if success_flag: success_parse_count += 1

        if "TokenRatio" in metric_names:
            page_texts.append(sample['pageTextClean'] if 'pageTextClean' in sample else sample['pageText'])

    ic(success_parse_count)
    metric2scores = {}
    metric2score = {}

    for metric_name in metric_names:
        theta = kwargs.get('theta', 0.2)
        if "F1" in metric_name:
            if "F1s" in metric_name:
                score_dict, scores_dict = doc_evaluate(metric=metric_name, targets=gts, predictions=preds, theta=theta)
            elif "CategorizedF1" in metric_name:
                score_dict, scores_dict = doc_evaluate(metric=metric_name, targets=cat_gts, predictions=cat_preds, theta=theta)
            else:
                score_dict, scores_dict = doc_evaluate(metric=metric_name, targets=gts, predictions=preds)
            metric2scores.update(scores_dict)
            metric2score.update(score_dict)
        elif "CategorizedRecall" in metric_name:
            score_dict, scores_dict = doc_evaluate(metric=metric_name, targets=cat_gts, predictions=preds, theta=theta)
            metric2scores.update(scores_dict)
            metric2score.update(score_dict)
        elif metric_name == 'TokenRatio':
            score, scores = doc_evaluate(metric=metric_name, targets=page_texts, predictions=preds)
            metric2scores[metric_name] = scores
            metric2score[metric_name]=round(score,2)
        else:
            score, scores = doc_evaluate(metric=metric_name, targets=gts, predictions=preds)
            metric2scores[metric_name] = scores
            metric2score[metric_name]=round(score,2)

    ic(metric2score)
    
    write_json(metric2score, save_path)

    save_csv_path = Path(summary_dir) / 'summary.csv'
    run_id = Path(result_path).stem
    append_metrics_to_dynamic_csv(metric2score, save_csv_path, run_id)

    if save_each_eval:
        save_each_path = save_path.replace('.json', '_metrics.json')
        eval_result = []
        for i in range(len(preds)):
            each_result = { 
                'metric2score': [{'metric':metric, 'score': scores[i]} for metric, scores in metric2scores.items()],
                'gt': gts[i],
                'pred': preds[i]}
            
            if any('F1s' in s for s in metric_names):
                each_result['string_gt'] = gts[i]
            
            if any('Categorized' in s for s in metric_names):
                each_result['cat_gt'] = cat_gts[i]
                if any("CategorizedF1" in s for s in metric_names):
                    each_result['cat_pred'] = cat_preds[i]

            if save_cat_eval:
                if 'categories' in gts_raw[i]:
                    each_result['categories'] = gts_raw[i]['categories']
                else:
                    each_result['categories'] = gts_raw[i]['data']['categories'] if 'data' in gts_raw[i] else []
            eval_result.append(each_result)
        write_json(eval_result, save_each_path)

    if save_cat_eval:
        save_cat_path = save_path.replace('.json', '_case_categories.json')
        metrics_summary = compute_category_metrics(gts_raw, metric2scores)
        ic(metrics_summary)
        write_json(metrics_summary, save_cat_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='text detection evaluation')
    parser.add_argument('--eval_name', type=str, help='the evaluation note for save path')
    parser.add_argument('--eval_metrics', nargs="+", type=str, help='the evaluation metrics')
    parser.add_argument('--pred_path', type=str, help='the directory path of model prediction')
    parser.add_argument('--gt_path', type=str, help='the directory path of ground truth')
    parser.add_argument('--save_dir', type=str, help='the directory to save the detailed evaluation result')
    parser.add_argument('--summary_dir', type=str, default='', help='the directory to save the summary of all eval results')
    parser.add_argument('--option', nargs="*", help="Optional key=value pairs", default=[])

    args = parser.parse_args()

    eval_name = args.eval_name
    eval_metrics = args.eval_metrics
    pred_path = args.pred_path
    gt_path = args.gt_path
    save_dir = args.save_dir
    summary_dir = args.summary_dir

    sample_num = -1
    theta = 0.2

    option_dict = dict(item.split("=", 1) for item in args.option)
    if 'sample_num' in option_dict:
        sample_num = int(option_dict['sample_num'])
    if 'theta' in option_dict:
        theta = float(option_dict['theta'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, eval_name+'_eval_result.json')

    if not os.path.exists(pred_path):
        print('not exists:', pred_path)
        exit(0)

    llm_text_detection_eval(
        metric_names=eval_metrics, 
        result_path=pred_path, 
        gt_path=gt_path, 
        save_path=save_path,
        summary_dir=summary_dir, 
        save_each_eval=True,
        save_cat_eval=True,
        sample_num=sample_num,
        theta=theta
    )

    print('==============================================')
    
