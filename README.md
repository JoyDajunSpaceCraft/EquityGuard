\documentclass{article}
\usepackage{hyperref}

\begin{document}

\title{EquityGuard: Enhancing Equity in Large Language Models for Medical Applications}
\date{}
\maketitle

\section*{Overview}

\textbf{EquityGuard} is a contrastive learning-based framework designed to detect and mitigate biases in Large Language Models (LLMs) used in healthcare applications. The framework addresses inequities observed in tasks such as clinical trial matching and medical question answering (MedQA), which are crucial for clinical decision support and translational research. By systematically disentangling sensitive attributes such as race, sex, and social determinants of health (SDOH), EquityGuard promotes fairer and more equitable healthcare outcomes.

\section*{Key Features}
\begin{itemize}
    \item \textbf{Bias Detection Mechanism}: Identifies and corrects unfair predictions in LLM-based systems.
    \item \textbf{Contrastive Learning}: Uses self-supervised techniques to align data representations, mitigating inequity by targeting biased inputs.
    \item \textbf{Task-Specific Implementation}: Applied to clinical trial matching and medical question answering tasks while maintaining high performance and fairness.
    \item \textbf{Extensive Evaluation}: Assessed on datasets such as SIGIR, TREC 2021, TREC 2022, MedQA, and MedMCQA using models like GPT-4, Gemini, and Claude.
\end{itemize}

\section*{Tasks}

\subsection*{Clinical Trial Matching}
EquityGuard automates the process of matching patients to appropriate clinical trials based on eligibility criteria from patient records and trial protocols. It minimizes bias related to race, gender, and other SDOH factors, ensuring equitable recruitment for clinical trials.

\subsection*{Medical Question Answering (MedQA)}
EquityGuard addresses inequities in LLMs used for medical question answering (Q\&A), ensuring fair responses across sensitive categories. By mitigating biases, the framework improves the accuracy and fairness of answers provided by LLMs in clinical decision support systems.

\section*{Datasets}
The framework was tested on the following datasets:
\begin{itemize}
    \item \textbf{SIGIR 2016}: Clinical trial descriptions from ClinicalTrials.gov and patient case reports.
    \item \textbf{TREC 2021 and 2022}: Datasets focusing on automating the clinical trial matching process.
    \item \textbf{MedQA}: A large-scale dataset containing medical questions from the Chinese medical licensing exam.
    \item \textbf{MedMCQA}: A multi-choice question-answering dataset based on medical topics from AIIMS and NEET PG exams.
\end{itemize}

\section*{Results}
EquityGuard was shown to improve fairness across sensitive attributes by minimizing the influence of race, gender, and SDOH in both trial matching and medical question answering tasks. Key improvements include:

\begin{itemize}
    \item \textbf{NDCG@10}: Improved fairness in trial matching, with consistent performance across different racial and socioeconomic groups.
    \item \textbf{Error Rate}: Reduced error rates in medical Q\&A, particularly in categories such as race and socioeconomic status.
    \item \textbf{Fairness Metrics}: Significant reductions in Equal Opportunity (EO) and Demographic Parity (DP) disparities, ensuring that predictions are not disproportionately influenced by sensitive attributes.
\end{itemize}

\section*{Installation}
To use EquityGuard, clone the repository and install the required dependencies:
\begin{verbatim}
git clone https://github.com/JoyDajunSpaceCraft/EquityGuard.git
cd EquityGuard
pip install -r requirements.txt
\end{verbatim}

\section*{Usage}
The framework can be applied to both clinical trial matching and medical question answering tasks. Sample scripts are provided for each task in the \texttt{scripts/} directory:
\begin{itemize}
    \item \texttt{run\_clinical\_trial\_matching.py}: Run the clinical trial matching task with EquityGuard.
    \item \texttt{run\_medqa.py}: Run the MedQA task with bias mitigation using EquityGuard.
\end{itemize}

Example usage:
\begin{verbatim}
python scripts/run_clinical_trial_matching.py --dataset sigir --model GPT-4
\end{verbatim}

\section*{License}
This project is licensed under the \href{LICENSE}{MIT License}.

\section*{Acknowledgements}
This research was supported by the University of Pittsburgh Momentum Funds and the National Institutes of Health. Special thanks to Qiao Jin, Yifan Yang, and Zhiyong Lu from the National Library of Medicine for their assistance in explaining the TrialGPT results.

\end{document}
