# EACL26-detect-latin
Official implementation and dataset of EACL 2026 paper 816: Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark.

## Introduction
This paper presents a novel task of extracting low-resourced and noisy Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary zero-shot models is achievable, yet these models lack a functional comprehension of Latin. This study establishes a comprehensive baseline for processing Latin within mixed-language corpora, supporting quantitative analysis in intellectual history and historical linguistics.

## 🎯 Problem Definition
Given a document page $D$, let $I_D$ denote its image and $T_D$ denote its OCR-processed text. A system must perform the following two subtasks:

* **Task 1 (Page-level Latin Detection):** Predict a binary label $y_D \in \{0,1\}$, where $y_D = 1$ indicates that the page contains at least one segment in Latin, and $y_D = 0$ otherwise.
* **Task 2 (Latin Segment Extraction):** If $y_D = 1$, extract a list of text spans $S_D = [s_1, s_2, \ldots, s_n]$, where each $s_i \in T_D$ is a contiguous Latin segment string.

## 📚 Data
In total, **724** pages were annotated, with **594** identified as containing Latin. 

We divided the annotated Latin segments into **12** language integration categories. Each category represents a specific way in which Latin is used in 18th-century British books and how it relates to English-language text:

| Category | Description |
| :--- | :--- |
| **Bilingual Editions** | Original Latin text and its English translation appearing right next to it (e.g., parallel columns or facing pages). |
| **Independent Latin Text** | Original Latin text by the author, sometimes accompanied by English text on the same page but structurally distinct. |
| **Direct Quotations** | Latin phrases or sentences quoted verbatim, often embedded within an otherwise predominantly English text. |
| **Code Switching** | Text where the writer alternates between Latin and English within the same sentence or paragraph, often for stylistic or rhetorical purposes. |
| **Dictionaries** | Latin text appearing in a dictionary-like context, such as entries defining individual Latin words with translations or explanations. |
| **Footnotes** | Latin text appearing in annotations or footnotes, often providing definitions, sources, or explanations for terms used in the main text. |
| **Emblematic Quotes** | Latin phrases used as symbolic or thematic elements (e.g., mottos, epigraphs, maxims), typically set apart from the main text. |
| **Sidenotes** | Printed or authorial notes placed in the margins or alongside the main text containing Latin. |
| **Legal Formulae** | Standardized Latin phrases or terminology used specifically in legal contexts. |
| **Ecclesiastical Formulae** | Standardized Latin expressions used in religious, liturgical, or ecclesiastical contexts. |
| **Tables and Charts** | Use of Latin in tabular data, genealogies, calendars, scientific diagrams, inflection tables, or mathematical charts. |
| **Indexes and Catalogs** | Use of Latin in structured lists such as indices, bibliographies, book catalogs, or errata lists. |

You can download the full dataset (images + annotations) at **[Zenodo](https://doi.org/10.5281/zenodo.18377924)**.

## 🤖 Model Outputs & Results
Coming soon...

## 📏 How to Run Scripts

### Prerequisites
- Our evaluation is based on a local `vLLM` server. Learn more from the [vLLM official doc](https://docs.vllm.ai/en/latest/).

- Before evaluation, please prepare a Python environment that satisfies `requirements.txt`. 
  - e.g. `pip install --user -r requirements.txt`

- To use the default path settings, simply place all data in the `data` directory in the project root.

### Run vLLM Inference
- Start the vLLM OpenAI API server:

    ```
    python -m vllm.entrypoints.openai.api_server --model <MODEL_NAME> --trust-remote-code 
    ```

    - Replace `<MODEL_NAME>` with your model (supported by vLLM), e.g., OpenGVLab/InternVL3-38B.
    - Keep the server running and open a new terminal to run inference.

- Run the inference script to generate predictions:

    ```
    python detect_vl_latin_async.py \
    --model_name <MODEL_NAME> \
    --test_name <TEST_NAME> \
    --modality <MODALITY> \
    --prompt "<PROMPT>" \
    --data_path <DATA_JSON_PATH> \
    --image_dir <IMAGE_DIR> \
    --output_dir <OUTPUT_DIR>
    ```

    - The output will be a JSON file like `<MODEL_NAME>_<TEST_NAME>.json` in `<OUTPUT_DIR>`.

### Run Evaluation
- Run the evaluation script to compute metrics:

    ```
    python text_detection_eval.py \
    --eval_name <EVAL_NAME> \
    --eval_metrics CaseF1 N1F1s TokenRatio N1CategorizedRecall \
    --pred_path <PRED_JSON_PATH> \
    --gt_path <GT_JSON_PATH> \
    --save_dir <SAVE_DIR> \
    --summary_dir <SUMMARY_DIR>
    ```

    - After evaluation, results will be saved in `<SAVE_DIR>` as JSON files with detailed metrics.

## 📝 TODO

◽️ Data content specification.

◽️ Model output files to be released.

◽️ vLLM inference pipeline testing.

◽️ Lingua baseline code.

◽️ Evaluation scripts further refactoring.

◽️ Acknowledgement.