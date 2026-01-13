---
title: Pickleball Predictor
sdk: docker
app_port: 8501
---

## General Description

This project implements a Logistic Regression classification model to predict the winner of a pickleball match based on player attributes and match conditions. The model outputs both a predicted winner and the associated probability of winning.


## Dataset Generation Approach

### Dataset Features 
Pickleball dataset was simulated for 200 matches along with player attributes and match conditions that was randomly assigned within predefined ranges about: 
- player's ranking (between 3.0 and 6.0)
- player's win rate (between 0.4 and 0.9) 
- court type (indoor or outdoor)
- weather (sunny, windy, cloudy, indoor was set to always be indoor_aircon) 
- match duration (between 45 and 75 minutes) 
- winner for each match

Player-related features were represented as differences between the two players (e.g. rank difference, win rate difference).

### Target Variable
The target variable is the match winner. In the simulated dataset, match outcomes were determined using a rule-based probability model. Starting with a base win probability which introduced weightage to players rank and win rate differences (player 1 - player 2) allowing for players with a higher rank/win rate to win the match: 

 base_win_prob = 0.5 + (rank_diff * 0.4) + (win_rate_diff * 0.3)

However, to further simulate real-life match dynamics, weather conditions and match duration were introduced as factors that influence winning probability. 

**Weather Effects:**
- Windy conditions: A 50% reduction in skill advantage is applied (weather_factor = 0.5), making matches more unpredictable
- Cloudy conditions: A 10% increase in skill advantage is applied (weather_factor = 1.1), favoring the better player
- Sunny/Indoor conditions: No effect (weather_factor = 1.0)

**Duration Effects:**
Longer match durations reduce the impact of skill differences, making matches more competitive. The duration factor is calculated as:

duration_normalized = (duration - 45) / (75 - 45)

duration_factor = 1.0 - (duration_normalized * 0.4)

This means that a 45-minute match has no duration effect (duration_factor = 1.0), while a 75-minute match reduces skill impact by up to 40% (duration_factor = 0.6), effectively narrowing the skill gap between players.

**The final winning probability formula is:**

win_prob = 0.5 + (base_win_prob - 0.5) * weather_factor * duration_factor
 


## Model Architecture 

The dataset was split into 80% for training+validation (160/200) and 20% (40/200) for the final hold-out test set. 

A scikit-learn pipeline was created which includes the preprocesser and model training, ensuring consistent transformations and preventing data leakage. 

### Preprocessing
Numerical features were standardised using StandardScaler. Categorical features were encoded using one-hot encoding. 

### Training and Validation
A stratified K-fold cross validation with 5 splits (each fold had 106/160 matches as the training set and 26/160 matches as the validation set). Accuracy score was computed for each fold and averaged across all folds. 

### Evaluation
The final pipeline was trained on the full training set and evaluated on the hold-out test set. The model outputs the prediction of the final winner and probability of winning between both players. 

Model performance was assesed through a confusion matrix, ROC-AUC curve and score, as well as a classification report (precision, recall, f1-score, accuracy) was generated. 

---

## Setup Instructions

### Option 1: Use the live app 

The application is deployed on Hugging Face Spaces and can be used directly in the browser. No installion or setup is required.

**Live Demo**: 
https://huggingface.co/spaces/tracycje/pickleball#predicted-winner-player-1

### Option 2: Run the app locally 

Follow the steps below to run the application on your local machine. 

1. Clone the repository
```bash
git clone https://github.com/tracycje/pickleball-winner-predictor.git
cd pickleball-winner-predictor 
```

2. Create and activate a virtual environment 

Windows: 
python -m venv venv

macOS / Linux: 
source venv/bin/activate

3. Install dependencies 
pip install -r requirements.txt

4. Run the streamlit app 
streamlit run app.py
The app will open in your browser at: 
http://localhost:8501


--- 


## Reflection: Scaling to Live Match Predicitons or Pro Player Analytics

### Summary / TL;DR: 
To scale this project to live match prediction or professional player analytics, the key challenges extend beyond model choice to data, time, and system design. A production system would require larger, higher-coverage datasets with richer player features and continuous updates to reflect evolving performance. More expressive models could capture complex, nonlinear interactions, but would require careful regularisation and class-balancing. Temporal modelling becomes essential with recent matches carrying more weight than older data. At the system level, real-time pipelines enable low-latency live predictions, while batch pipelines are better suited for efficient player analytics and reporting. Continuous evaluation, drift monitoring, calibration tracking, and scheduled retraining are critical to keep predictions reliable, fast, and aligned with real-world dynamics.

### Detailed Understanding (for personal learning): 

- Dataset: Current model performance is limited by data quality and coverage. For live match predictions and pro player analytics, a larger dataset is required with higher coverage across scenarios. More features may also be implemented e.g. previous injury, tournament, age etc. which increases the complexity of interactions 

- Model: Logistic regression is a conservative model. More expressive models like Gradient Boosting can capture non-linear interactions which produces better model performance when relationship between features become more complex. However, careful regularisation and class-balancing strategies is needed to prevent overfitting. 

- Temporal modelling: Live predictions would require continuous data updates (player tracking/live match feeds) leading to a growing dataset. Player performance also changes over time which means more weightage needs to applied on a player's recent match over their old ones. Rolling averages or time-decay features should be added. 

- Evaluation and monitoring: To account for temporal decay, we need to monitor model drift as the relationship between inputs/features may change over time. In pro-player analytics, we need to track feature distributions over time and raise alerts when they diverge (mean/variance shifts). Then, model has to be retrained periodically to ensure it is trained on the latest matches and refitted. Tracking calibration and not just accuracy. Calibration indicates when a model's confidence no longer matches the reality (i.e. player 1 has a predicted winning probability of 0.9 but the actual win rate is only 0.6 making the model "overconfident"). Calibration is an early warning for model drift. Calibration tools include reliability diagrams, brier score, expected calibration error (ECE).  

- Systems level scaling: In live-match predictions a real-time pipeline is needed which is more computationally expensive and more latency sensitive. However, it provides a more accurate mid-match winner prediction. For pro-player anayltics, a batch pipeline may be used to produce daily player reports or weekly predictions. This is more efficient, computationally cheaper and easier to scale. 

- Conclusion: Scaling to keep it reliable, fast, and aligned with changing real-world data. It also means model training must be scheduled and automated. Monitoring of feature interactions to keep model performing well. 



















