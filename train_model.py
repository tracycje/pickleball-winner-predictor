## Pickleball Winner ML Classification Model
# 1. Simulating pickleball match data
# 2. Splitting data: stratified k-fold validation, train-test split 
# 3. Feature engineering: player stats, match conditions
# 4. Pipeline: preprocessing(Scaling, Feature Selection), model training
# 5. Model evaluation: accuracy, precision, recall, F1-score
# 6. Hyperparameter tuning: Grid Search, Random Search
# 7. Select the best model based on evaluation metrics
# 8. Final model evaluation and saving 

# Install necessary libraries
# !pip install numpy pandas matplotlib scikit-learn

## Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import joblib

### SIMULATING PICKLEBALL DATASET ### 
# Setting seed for reproducibility
np.random.seed(11)

def generate_pickleball_data(num_matches=200): 
    data = []

    for i in range(num_matches):
        # Player ranking: p1_rank, p2_rank
        p1_rank = round(np.random.uniform(3.0, 6.0),1)
        p2_rank = round(np.random.uniform(3.0, 6.0),1)
        # Player win rate: p1_win_rate, p2_win_rate
        p1_win_rate = round(np.random.uniform(0.4, 0.9),2)
        p2_win_rate = round(np.random.uniform(0.4, 0.9),2)
        # Court type: court_type + Weather conditons 
        court_type = np.random.choice(['indoor', 'outdoor'])
        if court_type == 'indoor': 
            weather = 'indoor_aircon'
        else: 
            weather = np.random.choice(['sunny', 'windy', 'cloudy'])

        # Randomly swap player identities to remove Player 1 bias
        if np.random.rand() < 0.5:
            p1_rank, p2_rank = p2_rank, p1_rank
            p1_win_rate, p2_win_rate = p2_win_rate, p1_win_rate

        
        # Match duration: minutes 
        duration = np.random.randint(45, 75)

        ## Determining winner: player with higher rank has a higher chance of winning 
        # Calculate stat differences
        rank_diff = p1_rank - p2_rank
        win_rate_diff = p1_win_rate - p2_win_rate
        
        # Check if stats are effectively equal (within tolerance)
        if abs(rank_diff) < 0.01 and abs(win_rate_diff) < 0.01:
            # Equal stats: fair 50/50 coin flip using np.random.choice for true fairness
            winner = np.random.choice(['Player 1', 'Player 2'])   # !!! STILL UNSURE WHY IT WOULD STILL PREFER PLAYER 1 (maybe the way that it's been trained?)
        else:
            # Different stats: calculate win probability based on differences
            # This gives Player 1's win probability
            win_prob = 0.5 + (rank_diff * 0.4) + (win_rate_diff * 0.3)
            win_prob = max(0.01, min(0.99, win_prob))  # capping between 1% to 99%
            
            # if P1 rank > P2 rank --> positive value --> higher win prob for P1
            # if P1 rank < P2 rank --> negative value --> lower win prob for P1
            # *0.4 controls the influence of rank diff on win probability
            
            # Use random choice with probabilities for more accurate distribution
            winner = np.random.choice(['Player 1', 'Player 2'], p=[win_prob, 1 - win_prob]) 

        ## Appending match data
        data.append({ 
            'p1_rank': p1_rank,
            'p2_rank': p2_rank,
            'p1_win_rate': p1_win_rate,
            'p2_win_rate': p2_win_rate,
            'court_type': court_type,
            'weather': weather,
            'duration': duration,
            'winner': winner
        })

    return pd.DataFrame(data)


### SIMULATING THE PICKLEBALL DATASET ###
df = generate_pickleball_data(num_matches=200)
print(df.head(10))
# Calculate wins for each player 
win_counts = df['winner'].value_counts()
print(win_counts)

from sklearn.model_selection import train_test_split, StratifiedKFold

x = df.drop('winner', axis=1)  # axis 1 --> columns; we want everything except 'winner' 
y = df['winner']               # because winner is our target variable

### HOLD-OUT TEST SET ### 
# setting aside 20% of dataset as final test set 
# prevents data leakage and ensures unbiased evaluation
x_temp, x_test, y_temp, y_test = train_test_split(
    x, y, test_size =0.2, stratify=y, random_state=11
)

# checking size of datasets 
print(f"\nTOTAL MATCHES: {len(df)}")
print(f"TRAIN+VALIDATION SET: {len(x_temp)}")
print(f"HOLD-OUT TEST SET: {len(x_test)}")

### LOGISTIC REGRESSION ### 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

### PREPROCESSOR ###

# Identify column types 
num_features = ['p1_rank', 'p2_rank', 'p1_win_rate', 'p2_win_rate', 'duration']
cat_features = ['court_type', 'weather']

# Create a preprocessor to handle both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder( handle_unknown='ignore'), cat_features)
    ]) 


### LOGISTIC REGRESSION MODEL ###

logreg = LogisticRegression( 
    solver="lbfgs",
    max_iter=1000,
    C=1,
    class_weight=None,
    random_state=11)


### PIPELINE: PREPROCESSOR + MODEL ###

logreg_pipe = make_pipeline(preprocessor, logreg)


### STRATIFIED K-FOLD CROSS-VALIDATION ###

# using the remaining 80% (train+val) for stratified k-fold cv
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
print("\nStratified K-Fold Splits:")
for fold, (train_index, val_index) in enumerate(skf.split(x_temp, y_temp)):
    x_train, x_val = x_temp.iloc[train_index], x_temp.iloc[val_index]
    y_train, y_val = y_temp.iloc[train_index], y_temp.iloc[val_index]
    print(f"\nFold {fold+1}:")
    print(f"Training set: {len(x_train)} matches")
    print(f"Validation set: {len(x_val)} matches")

# Evaluation based on CV
cv_scores = cross_val_score(logreg_pipe, x_train, y_train, cv=skf, scoring='accuracy')
print(f"\nScores for each fold: {cv_scores.round(4)} \
      \nMean CV Accuracy: {cv_scores.mean():.4f}")


### FIT MODEL ON HOLD-OUT TEST SET###

logreg_pipe.fit(x_temp, y_temp)

# Saving trained pipeline 
joblib.dump(logreg_pipe, "pickleball_winner_model.joblib")

# Prdicting winner and probabilities on hold-out test set 
y_test_prob = logreg_pipe.predict_proba(x_test)[:,1]
y_test_pred = logreg_pipe.predict(x_test)


### EVALUATION ON FINAL TEST SET ###
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder

# Confusion Matrix
labels = ['Player 1', 'Player 2']
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot()

# ROC-AUC Curve
le = LabelEncoder()
y_temp_bin = le.fit_transform(y_temp)    # converting labels to binary 
y_test_bin = le.transform(y_test)
auc_score = roc_auc_score(y_test, y_test_prob)
print(f"\nROC-AUC Score on Test Set: {auc_score:.4f}")
display_roc = RocCurveDisplay.from_predictions(y_test, 
                                               y_test_prob,
                                               pos_label='Player 2')
                                        

print("\n--- Final Test Set Evaluation ---")
print(classification_report(y_test, y_test_pred, target_names=['Player 1', 'Player 2']))



## Pickleball Winner ML Classification Model
# 1. Simulating pickleball match data
# 2. Splitting data: stratified k-fold validation, train-test split 
# 3. Feature engineering: player stats, match conditions
# 4. Pipeline: preprocessing(Scaling, Feature Selection), model training
# 5. Model evaluation: accuracy, precision, recall, F1-score
# 6. Hyperparameter tuning: Grid Search, Random Search
# 7. Select the best model based on evaluation metrics
# 8. Final model evaluation and saving 

# Install necessary libraries
# !pip install numpy pandas matplotlib scikit-learn

## Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import joblib

### SIMULATING PICKLEBALL DATASET ### 
# Setting seed for reproducibility
np.random.seed(11)

def generate_pickleball_data(num_matches=200): 
    data = []

    for i in range(num_matches):
        # Player ranking: p1_rank, p2_rank
        p1_rank = round(np.random.uniform(3.0, 6.0),1)
        p2_rank = round(np.random.uniform(3.0, 6.0),1)
        # Player win rate: p1_win_rate, p2_win_rate
        p1_win_rate = round(np.random.uniform(0.4, 0.9),2)
        p2_win_rate = round(np.random.uniform(0.4, 0.9),2)
        # Court type: court_type + Weather conditons 
        court_type = np.random.choice(['indoor', 'outdoor'])
        if court_type == 'indoor': 
            weather = 'indoor_aircon'
        else: 
            weather = np.random.choice(['sunny', 'windy', 'cloudy'])

        # Randomly swap player identities to remove Player 1 bias
        if np.random.rand() < 0.5:
            p1_rank, p2_rank = p2_rank, p1_rank
            p1_win_rate, p2_win_rate = p2_win_rate, p1_win_rate

        
        # Match duration: minutes 
        duration = np.random.randint(45, 75)

        ## Determining winner: player with higher rank has a higher chance of winning 
        # Calculate stat differences
        rank_diff = p1_rank - p2_rank
        win_rate_diff = p1_win_rate - p2_win_rate
        
        # Check if stats are effectively equal (within tolerance)
        if abs(rank_diff) < 0.01 and abs(win_rate_diff) < 0.01:
            # Equal stats: fair 50/50 coin flip using np.random.choice for true fairness
            winner = np.random.choice(['Player 1', 'Player 2'])
        else:
            # Different stats: calculate win probability based on differences
            # This gives Player 1's win probability
            win_prob = 0.5 + (rank_diff * 0.4) + (win_rate_diff * 0.3)
            win_prob = max(0.01, min(0.99, win_prob))  # capping between 1% to 99%
            
            # if P1 rank > P2 rank --> positive value --> higher win prob for P1
            # if P1 rank < P2 rank --> negative value --> lower win prob for P1
            # *0.4 controls the influence of rank diff on win probability
            
            # Use random choice with probabilities for more accurate distribution
            winner = np.random.choice(['Player 1', 'Player 2'], p=[win_prob, 1 - win_prob]) 

        ## Appending match data
        data.append({ 
            'p1_rank': p1_rank,
            'p2_rank': p2_rank,
            'p1_win_rate': p1_win_rate,
            'p2_win_rate': p2_win_rate,
            'court_type': court_type,
            'weather': weather,
            'duration': duration,
            'winner': winner
        })

    return pd.DataFrame(data)


### SIMULATING THE PICKLEBALL DATASET ###
df = generate_pickleball_data(num_matches=200)
print(df.head(10))
# Calculate wins for each player 
win_counts = df['winner'].value_counts()
print(win_counts)

from sklearn.model_selection import train_test_split, StratifiedKFold

x = df.drop('winner', axis=1)  # axis 1 --> columns; we want everything except 'winner' 
y = df['winner']               # because winner is our target variable

### HOLD-OUT TEST SET ### 
# setting aside 20% of dataset as final test set 
# prevents data leakage and ensures unbiased evaluation
x_temp, x_test, y_temp, y_test = train_test_split(
    x, y, test_size =0.2, stratify=y, random_state=11
)

# checking size of datasets 
print(f"\nTOTAL MATCHES: {len(df)}")
print(f"TRAIN+VALIDATION SET: {len(x_temp)}")
print(f"HOLD-OUT TEST SET: {len(x_test)}")

### LOGISTIC REGRESSION ### 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

### PREPROCESSOR ###

# Identify column types 
num_features = ['p1_rank', 'p2_rank', 'p1_win_rate', 'p2_win_rate', 'duration']
cat_features = ['court_type', 'weather']

# Create a preprocessor to handle both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder( handle_unknown='ignore'), cat_features)
    ]) 


### LOGISTIC REGRESSION MODEL ###

logreg = LogisticRegression( 
    solver="lbfgs",
    max_iter=1000,
    C=1,
    class_weight=None,
    random_state=11)


### PIPELINE: PREPROCESSOR + MODEL ###

logreg_pipe = make_pipeline(preprocessor, logreg)


### STRATIFIED K-FOLD CROSS-VALIDATION ###

# using the remaining 80% (train+val) for stratified k-fold cv
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
print("\nStratified K-Fold Splits:")
for fold, (train_index, val_index) in enumerate(skf.split(x_temp, y_temp)):
    x_train, x_val = x_temp.iloc[train_index], x_temp.iloc[val_index]
    y_train, y_val = y_temp.iloc[train_index], y_temp.iloc[val_index]
    print(f"\nFold {fold+1}:")
    print(f"Training set: {len(x_train)} matches")
    print(f"Validation set: {len(x_val)} matches")

# Evaluation based on CV
cv_scores = cross_val_score(logreg_pipe, x_train, y_train, cv=skf, scoring='accuracy')
print(f"\nScores for each fold: {cv_scores.round(4)} \
      \nMean CV Accuracy: {cv_scores.mean():.4f}")


### FIT MODEL ON HOLD-OUT TEST SET###

logreg_pipe.fit(x_temp, y_temp)

# Saving trained pipeline 
joblib.dump(logreg_pipe, "pickleball_winner_model.joblib")

# Prdicting winner and probabilities on hold-out test set 
y_test_prob = logreg_pipe.predict_proba(x_test)[:,1]
y_test_pred = logreg_pipe.predict(x_test)


### EVALUATION ON FINAL TEST SET ###
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder

# Confusion Matrix
labels = ['Player 1', 'Player 2']
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot()

# ROC-AUC Curve
le = LabelEncoder()
y_temp_bin = le.fit_transform(y_temp)    # converting labels to binary 
y_test_bin = le.transform(y_test)
auc_score = roc_auc_score(y_test, y_test_prob)
print(f"\nROC-AUC Score on Test Set: {auc_score:.4f}")
display_roc = RocCurveDisplay.from_predictions(y_test, 
                                               y_test_prob,
                                               pos_label='Player 2')
                                        

print("\n--- Final Test Set Evaluation ---")
print(classification_report(y_test, y_test_pred, target_names=['Player 1', 'Player 2']))


