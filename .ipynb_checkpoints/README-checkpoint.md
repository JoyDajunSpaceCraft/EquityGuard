# EquityGuard: Enhancing Equity in Large Language Models for Medical Applications

## Overview

**EquityGuard** is a contrastive learning-based framework designed to detect and mitigate biases in Large Language Models (LLMs) used in healthcare applications. The framework addresses inequities observed in tasks such as clinical trial matching (CTM) and medical question answering (MQA), which are crucial for clinical decision support and translational research. By systematically disentangling sensitive attributes such as race, sex, and social determinants of health (SDOH), EquityGuard promotes fairer and more equitable healthcare outcomes.
dd
## Key Features

- **Bias Detection Mechanism**: Identifies and corrects unfair predictions in LLM-based systems.
- **Contrastive Learning**: Uses self-supervised techniques to align data representations, mitigating inequity by targeting biased inputs.
- **Task-Specific Implementation**: Applied to clinical trial matching and medical question-answering tasks while maintaining high performance and fairness.
- **Extensive Evaluation**: Assessed on SIGIR, TREC 2021, TREC 2022, MedQA, and MedMCQA using models like GPT-4, Gemini, and Claude.

## Tasks

### Clinical Trial Matching

EquityGuard automates matching patients to appropriate clinical trials based on eligibility criteria from patient records and trial protocols. It minimizes bias related to race, gender, and other SDOH factors, ensuring equitable recruitment for clinical trials.

### Medical Question Answering (MedQA)

EquityGuard addresses inequities in LLMs used for medical question answering (Q&A), ensuring fair responses across sensitive categories. By mitigating biases, the framework improves the accuracy and fairness of answers provided by LLMs in clinical decision support systems.

## Datasets

The framework was tested on the following datasets:

- **SIGIR 2016**: Clinical trial descriptions from ClinicalTrials.gov and patient case reports.
- **TREC 2021 and 2022**: Datasets focusing on automating the clinical trial matching process.
- **MedQA**: A large-scale dataset containing medical questions from the Chinese medical licensing exam.
- **MedMCQA**: A multi-choice question-answering dataset based on medical topics from AIIMS and NEET PG exams.

## Results

EquityGuard was shown to improve fairness across sensitive attributes by minimizing the influence of race, gender, and SDOH in both trial matching and medical question answering tasks. Key improvements include:

- **NDCG@10**: Improved fairness in trial matching, with consistent performance across different racial and socioeconomic groups.
- **Error Rate**: Reduced error rates in medical Q&A, particularly in categories such as race and socioeconomic status.
- **Fairness Metrics**: Significant reductions in Equal Opportunity (EO) and Demographic Parity (DP) disparities, ensuring that predictions are not disproportionately influenced by sensitive attributes.

## Installation

To use EquityGuard, clone the repository and install the required dependencies:

```bash
git clone https://github.com/JoyDajunSpaceCraft/EquityGuard.git
cd EquityGuard
pip install -r requirements.txt
```
## Usage

The framework can be applied to both clinical trial matching and medical question answering tasks. Sample scripts are provided for each task in the `scripts/` directory:

- `run_clinical_trial_matching.py`: Run the clinical trial matching task with EquityGuard.
- `run_medqa.py`: Run the MedQA task with bias mitigation using EquityGuard.

Example usage for clinical trial matching:

```bash
python scripts/run_clinical_trial_matching.py --dataset sigir --model GPT-4
```
Example usage for MedQA:

```bash
python scripts/run_medqa.py --dataset medqa --model GPT-4
```
Make sure to configure the dataset paths and model checkpoints in the configuration file before running the scripts.
