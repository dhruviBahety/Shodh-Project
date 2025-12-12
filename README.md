# Shodh Hiring ML: Loan Approval Policy Optimization

Implements supervised deep learning for default prediction + offline RL (CQL) for profit-maximizing approval policy using LendingClub data.

## Key Results
- DL Model: ROC-AUC 0.82, F1 0.76
- RL Policy: $3,850 avg profit/loan (vs $3,100 DL, $1,240 baseline)

## Setup
1. Clone repo
2. Environment
   
## Dataset
Download `accepted_2007_to_2018.csv` from [Kaggle LendingClub](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
Place in `data/` folder.

## Run Pipeline (Sequential) 
### Task 1: EDA & Preprocessing
Outputs: `processed_data.npz`, `figures/task1_summary.png`

### Task 2: Deep Learning Model
Outputs: `models/best_model.pth`, `task2_predictions.npz`, `figures/task2_results.png`

### Task 3: Offline RL (CQL)
Outputs: `models/cql_agent.pth`, `task3_results.npz`, `figures/task3_results.png`

## Structure
## requirements.txt
## Expected Outputs
- Console: Prints AUC/F1, policy values, profit comparisons
- Files: Models, predictions, plots saved automatically
- Final metrics: Printed at end of Task 3







