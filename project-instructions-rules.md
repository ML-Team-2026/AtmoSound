# PAML Final Project Proposal
89 points

# Overview
In final projects, students will propose a machine learning application that addresses a real-world problem. Students will select a well-justified problem domain (e.g., housing, healthcare, social media), identify a publicly available dataset for training and testing a machine learning model, identify a machine learning model to solve the problem, and propose standard evaluate metrics to evaluate model performance. Then, students will build a Streamlit application that deploys the trained model. The objectives of final projects include:
- Choose a problem to address using machine learning.
- Identify and justify the machine learning algorithms to solve the problem.
- Propose 2 or more machine learning algorithms for comparison.
- Identify 3 or more evaluation metrics to measure the performance of the algorithms.
- Conduct experiments using the machine learning models and evaluation metrics to identify the best performing algorithm.
- Build an online web application using Streamlit that deploys the best performing model.
 
# Final Project Proposal Outline
---
# 0.  Title (1 point)
Propose a project title that effectively describes the project goals.

---
# 1.  Abstract (6 points)
- Introduction/Background: Briefly introduce the topic and its significance, highlighting the problem or question your project addresses. (1 point)
- Purpose: State the main objective of your project. (1 point)
- Methods: Briefly describe two or more machine learning methods and three or more metrics used in your project. (2 point)
- Expected Results: Present the expected key findings of your project, focusing on the most important outcomes. (1 point)
- Significance: Discuss the implications or impact of your project, emphasizing its relevance and contribution to the field. (1 point)

---
# 2.  Introduction (7 points)
- Establish the problem or issue you want to solve:
1) Highlight the importance of the problem/issue to solve (Why should we care about this problem?; 1 point) and
2) Provide an overview of existing work on the problem. (1 point)
- Summarize prior efforts to solve the problem:
1) Identify gaps or unresolved issues in existing research that your project can fill or identify a technical focus that will be useful. (1 point)
- Summarize the proposed approach:
1) State and justify the proposed solution including 2 or more machine learning methods (1 point),
2) Indicate 3 or more evaluation metrics to measure success of solving the proposed problem (1 point), and
3) Provide details about the proposed Streamlit application and its intended use by a target user group (1 point).
- Broader Impact of the proposed project: What will the machine learning application contribute to the field (and beyond)? (1 point)

---
# 3.  Background (13 points)
- Comparisons to prior work: Discuss and cite the existing level of research done on the project topic or on related topics in the field to set context for your project. Be concise and mention only the relevant part of studies, ideally in chronological order to reflect the progress being made. (5 points)
- Share common knowledge: Highlight disputes in the field as well as claims made by scientists, organizations, or key stakeholders that need to be investigated. This forms the foundation of your methodology and solidifies the aims of your project. (5 points)
- Identify knowledge gaps: Describe if and how the methods and techniques used in the project are different from those used in previous research on similar topics. Based on prior work, describe how you anticipate your proposed approach will compare to prior approaches. (3 points)
Cite relevant literature using bibtex (e.g., on Google Scholar). Enter the bibtex code in the references.bib file in Overleaf and cite the source (e.g., \cite{example_bibtex_label}).

---
# 4.  End-to-End ML Pipeline
## 4.1  Offline Model Training and Evaluation
Project proposals must describe plans for offline development of machine learning algorithms to solve the proposed problem. Offline development is expected to be performed in Jupyter Notebooks but students are welcome to perform experiments in the Streamlit application. Explicitly state plans for offline development of machine learning models and experiments.

### 4.1.1  Data Collection, Exploration & Processing (6 points): 
1) Describe the dataset(s) you plan to use for training machine learning models including data types (e.g., numerical, categorical, image, text, and audio data; 1 point),
2) quantity of data (1 point),
3) year it was released (if available; 1 point)),
4) source of the dataset (e.g., Kaggle; 1 point), and
5) how you plan to use it in the project (1 point).
6) State whether ground truth labels are provided and how you plan to use them (1 point). If there are none, state how you plan to generate annotations or provide justification for the planned evaluation procedure. At least one dataset is required.
7) Discuss data exploration displays or tasks associated with the website include figures/plots such as Scatterplots, Histograms, Boxplots, Lineplots, or others. (5 points)
8) State the preprocessing techniques used to prepare the dataset for ML algorithms including data cleaning (e.g., handling missing data), feature encoding (one-hot, integer), feature normalization, correlations, outlier removal, dimensionality reduction, feature creation, discretization, binarization, and augmentation. (5 points)
Provide justification for data exploration and preprocessing techniques and how expected findings are used in the next steps of the ML pipeline.

### 4.1.2  Methods and Model Training (13 points): 
1) Describe 2 or more machine learning algorithms to solve the proposed problem including problem formulation, relevant equations, and theoretical underpinnings (5 points). Options of machine learning algorithms include regression, classification, clustering, artificial neural networks, and Agentic AI (NO exceptions).
At least 2 machine learning methods are required. All models must be implemented without using high-level ML libraries (e.g., scikit-learn, XGBoost, TensorFlow, PyTorch training APIs). Use of NumPy and Pandas for numerical operations and preprocessing is permitted (NO exceptions). (2 points)
2) Justify why the machine learning algorithms are well-suited to solve the problem. (2 points)
3-4) Describe model inputs and outputs. Teams using ML libraries will be penalized for using ML libraries. Agentic AI frameworks are allowed. (4 points)

### 4.1.3  Model Evaluation (5 points): 
1) Describe at least three evaluation metrics to measure the success of solving the proposed problem including equations (1 point).
2)  Provide justification for why the evaluation measures are well-suited to address the problem (1 point).
3)  Describe the training and validation procedure including the train/test split (1 point).
4)  Describe the planned experiments including fine-tuning model hyperparameters (1 point).
5)  Discuss the methods used to avoid model overfitting and underfitting. (1 point)
Make sure that the experimental plan is described in enough detail to reproduce your results.

### 4.1.4  Model Deployment (5 points): 
Describe and justify how the deployed model will be chosen. The machine learning algorithms may have tradeoffs in performance in terms of the selected evaluation metrics. Provide details about your plans to select a model given these trade-offs.

## 4.2  Front-End Application Using Streamlit (18 points)
Discuss of the proposed web-based application development using the Streamlit library.
1) Describe the target population (2 points),
2) user interface (UI) layout (using sketches; 2 points),
3) (3-4) user inputs (using menus, buttons, etc) and expected interface outputs (4 points).
4) (5) Provide figures or sketches to depict the proposed website layout and provide detailed discussion on how the front- and back-end are connected (e.g., deploy a model by pressing a button). Design a clear, concise, and well-structured web application. Describe the attended application in detail. (10 points)

---
# 5.  Risk & Mitigation
1) Identify anticipated challenges (2 points), and
2) how the team plans to mitigate them e.g., programming, model training, and evaluation (2 points).

---
# 6.  Expected Outcomes
State the 2-3 expected outcomes of the project based on prior work or high-level intuitions of existing knowledge about the problem ( 5points).

---
# 7.  Team Member Contribution
Each team member is expected to contribute to the final project. Task responsibilities must be clearly stated in terms of offline model training and evaluation, and front-end development.
Technical Components: Include the team contributions in terms of technical goals of the project such as programming offline, front- and back-end components and algorithms. (5 points)
Writing Components: Include the team contributions for writing sections of the final report for all sections. (5 points)

---
# 8.  Proposal Writing, Formating, and Figures
## 8.1  Writing Clarity
Written proposals will be evaluated based on clarity, significance of the problem, and the technical quality of the work (the level of depth in the experimental analyses, does the approach make sense to solve the problem, are the algorithms explained in enough detail, etc).
Remember this is a formal document. It should have correct grammar and should be spell checked. It should have figures/tables that are clear, easy to read, and are referenced in the text.
It should adopt a formal writing tone - avoid slang, use active voice, use plural first person (“we” or “our team”), clear, and concise.
Finally, recall the writing rule for paragraph structure (you probably learned in grade school):
One main idea + three supporting sentences. If it gets beyond that, it's time for a new paragraph. Submit the proposal via Gradescope and make sure to submit as a team.
If you would like writing help, please try these resources at Cornell.

---
## 8.2  Figures/Tables
Include and label all your figures and tables and include captions to describe them. This should include: title, axes, units, legend (if there is more than one plot in the figure).

---
## 8.3  Format
The project proposals require the following format using this IEEE Overleaf template:
1) Double column
2) Single-space
3) 1-inch margin
4) 12 pt font size
5) 2 to 4 pages (excluding references)
# PAML Final Project Proposal
89 points

# Overview
In final projects, students will propose a machine learning application that addresses a real-world problem. Students will select a well-justified problem domain (e.g., housing, healthcare, social media), identify a publicly available dataset for training and testing a machine learning model, identify a machine learning model to solve the problem, and propose standard evaluate metrics to evaluate model performance. Then, students will build a Streamlit application that deploys the trained model. The objectives of final projects include:
- Choose a problem to address using machine learning.
- Identify and justify the machine learning algorithms to solve the problem.
- Propose 2 or more machine learning algorithms for comparison.
- Identify 3 or more evaluation metrics to measure the performance of the algorithms.
- Conduct experiments using the machine learning models and evaluation metrics to identify the best performing algorithm.
- Build an online web application using Streamlit that deploys the best performing model.
 
# Final Project Proposal Outline
---
# 0.  Title (1 point)
Propose a project title that effectively describes the project goals.

---
# 1.  Abstract (6 points)
- Introduction/Background: Briefly introduce the topic and its significance, highlighting the problem or question your project addresses. (1 point)
- Purpose: State the main objective of your project. (1 point)
- Methods: Briefly describe two or more machine learning methods and three or more metrics used in your project. (2 point)
- Expected Results: Present the expected key findings of your project, focusing on the most important outcomes. (1 point)
- Significance: Discuss the implications or impact of your project, emphasizing its relevance and contribution to the field. (1 point)

---
# 2.  Introduction (7 points)
- Establish the problem or issue you want to solve:
1) Highlight the importance of the problem/issue to solve (Why should we care about this problem?; 1 point) and
2) Provide an overview of existing work on the problem. (1 point)
- Summarize prior efforts to solve the problem:
1) Identify gaps or unresolved issues in existing research that your project can fill or identify a technical focus that will be useful. (1 point)
- Summarize the proposed approach:
1) State and justify the proposed solution including 2 or more machine learning methods (1 point),
2) Indicate 3 or more evaluation metrics to measure success of solving the proposed problem (1 point), and
3) Provide details about the proposed Streamlit application and its intended use by a target user group (1 point).
- Broader Impact of the proposed project: What will the machine learning application contribute to the field (and beyond)? (1 point)

---
# 3.  Background (13 points)
- Comparisons to prior work: Discuss and cite the existing level of research done on the project topic or on related topics in the field to set context for your project. Be concise and mention only the relevant part of studies, ideally in chronological order to reflect the progress being made. (5 points)
- Share common knowledge: Highlight disputes in the field as well as claims made by scientists, organizations, or key stakeholders that need to be investigated. This forms the foundation of your methodology and solidifies the aims of your project. (5 points)
- Identify knowledge gaps: Describe if and how the methods and techniques used in the project are different from those used in previous research on similar topics. Based on prior work, describe how you anticipate your proposed approach will compare to prior approaches. (3 points)
Cite relevant literature using bibtex (e.g., on Google Scholar). Enter the bibtex code in the references.bib file in Overleaf and cite the source (e.g., \cite{example_bibtex_label}).

---
# 4.  End-to-End ML Pipeline
## 4.1  Offline Model Training and Evaluation
Project proposals must describe plans for offline development of machine learning algorithms to solve the proposed problem. Offline development is expected to be performed in Jupyter Notebooks but students are welcome to perform experiments in the Streamlit application. Explicitly state plans for offline development of machine learning models and experiments.

### 4.1.1  Data Collection, Exploration & Processing (6 points): 
1) Describe the dataset(s) you plan to use for training machine learning models including data types (e.g., numerical, categorical, image, text, and audio data; 1 point),
2) quantity of data (1 point),
3) year it was released (if available; 1 point)),
4) source of the dataset (e.g., Kaggle; 1 point), and
5) how you plan to use it in the project (1 point).
6) State whether ground truth labels are provided and how you plan to use them (1 point). If there are none, state how you plan to generate annotations or provide justification for the planned evaluation procedure. At least one dataset is required.
7) Discuss data exploration displays or tasks associated with the website include figures/plots such as Scatterplots, Histograms, Boxplots, Lineplots, or others. (5 points)
8) State the preprocessing techniques used to prepare the dataset for ML algorithms including data cleaning (e.g., handling missing data), feature encoding (one-hot, integer), feature normalization, correlations, outlier removal, dimensionality reduction, feature creation, discretization, binarization, and augmentation. (5 points)
Provide justification for data exploration and preprocessing techniques and how expected findings are used in the next steps of the ML pipeline.

### 4.1.2  Methods and Model Training (13 points): 
1) Describe 2 or more machine learning algorithms to solve the proposed problem including problem formulation, relevant equations, and theoretical underpinnings (5 points). All models must be implemented without using high-level ML libraries (e.g., scikit-learn, XGBoost, TensorFlow, PyTorch training APIs). Use of NumPy and Pandas for numerical operations and preprocessing is permitted (NO exceptions). (Options of machine learning algorithms include regression, classification, clustering, artificial neural networks, and Agentic AI (NO exceptions).
At least 2 machine learning methods are required. 
- **All models must be implemented without using high-level ML libraries (e.g., scikit-learn, XGBoost, TensorFlow, PyTorch training APIs). Use of NumPy and Pandas for numerical operations and preprocessing is permitted (NO exceptions). (2 points)**
2) Justify why the machine learning algorithms are well-suited to solve the problem. (2 points)
3-4) Describe model inputs and outputs. Teams using ML libraries will be penalized for using ML libraries. Agentic AI frameworks are allowed. (4 points)
- **Teams using ML libraries will be penalized for using ML libraries.**

### 4.1.3  Model Evaluation (5 points): 
1) Describe at least three evaluation metrics to measure the success of solving the proposed problem including equations (1 point).
2)  Provide justification for why the evaluation measures are well-suited to address the problem (1 point).
3)  Describe the training and validation procedure including the train/test split (1 point).
4)  Describe the planned experiments including fine-tuning model hyperparameters (1 point).
5)  Discuss the methods used to avoid model overfitting and underfitting. (1 point)
Make sure that the experimental plan is described in enough detail to reproduce your results.

### 4.1.4  Model Deployment (5 points): 
Describe and justify how the deployed model will be chosen. The machine learning algorithms may have tradeoffs in performance in terms of the selected evaluation metrics. Provide details about your plans to select a model given these trade-offs.

## 4.2  Front-End Application Using Streamlit (18 points)
Discuss of the proposed web-based application development using the Streamlit library.
1) Describe the target population (2 points),
2) user interface (UI) layout (using sketches; 2 points),
3) (3-4) user inputs (using menus, buttons, etc) and expected interface outputs (4 points).
4) (5) Provide figures or sketches to depict the proposed website layout and provide detailed discussion on how the front- and back-end are connected (e.g., deploy a model by pressing a button). Design a clear, concise, and well-structured web application. Describe the attended application in detail. (10 points)

---
# 5.  Risk & Mitigation
1) Identify anticipated challenges (2 points), and
2) how the team plans to mitigate them e.g., programming, model training, and evaluation (2 points).

---
# 6.  Expected Outcomes
State the 2-3 expected outcomes of the project based on prior work or high-level intuitions of existing knowledge about the problem ( 5points).

---
# 7.  Team Member Contribution
Each team member is expected to contribute to the final project. Task responsibilities must be clearly stated in terms of offline model training and evaluation, and front-end development.
Technical Components: Include the team contributions in terms of technical goals of the project such as programming offline, front- and back-end components and algorithms. (5 points)
Writing Components: Include the team contributions for writing sections of the final report for all sections. (5 points)

---
# 8.  Proposal Writing, Formating, and Figures
## 8.1  Writing Clarity
Written proposals will be evaluated based on clarity, significance of the problem, and the technical quality of the work (the level of depth in the experimental analyses, does the approach make sense to solve the problem, are the algorithms explained in enough detail, etc).
Remember this is a formal document. It should have correct grammar and should be spell checked. It should have figures/tables that are clear, easy to read, and are referenced in the text.
It should adopt a formal writing tone - avoid slang, use active voice, use plural first person (“we” or “our team”), clear, and concise.
Finally, recall the writing rule for paragraph structure (you probably learned in grade school):
One main idea + three supporting sentences. If it gets beyond that, it's time for a new paragraph. Submit the proposal via Gradescope and make sure to submit as a team.
If you would like writing help, please try these resources at Cornell.

---
## 8.2  Figures/Tables
Include and label all your figures and tables and include captions to describe them. This should include: title, axes, units, legend (if there is more than one plot in the figure).
