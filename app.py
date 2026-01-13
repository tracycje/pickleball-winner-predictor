# Building an interface where users can input match conditions and receive a winner prediction
# Allow visual comparison of predicted vs actual outcomes on sample data

import streamlit as st
import pandas as pd
import joblib    # joins trained ML pipeline from disk
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

# Header image
st.image("https://transcode-v2.app.engoo.com/image/fetch/f_auto,c_lfill,w_300,dpr_3/https://assets.app.engoo.com/organizations/5d2656f1-9162-461d-88c7-b2505623d8cb/images/33XnsFuZKOrTbgkCbqx62J.jpeg", use_container_width=True)

# Sets browser tab title + centered layout 
st.set_page_config(page_title="🥒 Pickleball Winner Predictor", layout='centered')

# Loads the trained ML model from disk only ONCE 
@st.cache_resource
def load_model():
    return joblib.load("pickleball_winner_model.joblib")

model = load_model()

# Function to regenerate the original simulated dataset (same as pickleball.py)
@st.cache_data
def generate_pickleball_data(num_matches=200):
    """Generate pickleball match data - same function as in pickleball.py"""
    np.random.seed(11)  # Same seed as original
    data = []
    
    for i in range(num_matches):
        # Player ranking: p1_rank, p2_rank
        p1_rank = round(np.random.uniform(3.0, 6.0), 1)
        p2_rank = round(np.random.uniform(3.0, 6.0), 1)
        # Player win rate: p1_win_rate, p2_win_rate
        p1_win_rate = round(np.random.uniform(0.4, 0.9), 2)
        p2_win_rate = round(np.random.uniform(0.4, 0.9), 2)
        # Court type: court_type + Weather conditions 
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
            # Equal stats: fair 50/50 coin flip
            winner = np.random.choice(['Player 1', 'Player 2'])
        else:
            # Different stats: calculate base win probability based on differences
            # This gives Player 1's win probability
            base_win_prob = 0.5 + (rank_diff * 0.4) + (win_rate_diff * 0.3)
            
            # Apply weather effects: windy conditions add unpredictability
            # Windy weather reduces the influence of skill differences (moves prob closer to 0.5)
            weather_factor = 1.0  # default: no weather effect
            if weather == 'windy':
                weather_factor = 0.5  # windy reduces skill advantage by 50%
            elif weather == 'cloudy':
                weather_factor = 1.1  # cloudy increases skill advantage by 10%
            # sunny and indoor_aircon have no effect (weather_factor = 1.0)
            
            # Apply duration effects: longer matches are more competitive
            # Longer matches reduce skill difference impact (moves prob closer to 0.5)
            # Duration range: 45-75 minutes, normalize to 0-1 scale
            duration_normalized = (duration - 45) / (75 - 45)  # 0 to 1
            duration_factor = 1.0 - (duration_normalized * 0.4)  # longer matches reduce skill impact by up to 40%
            
            # Apply modifiers: move win_prob closer to 0.5 based on weather and duration
            # Formula: adjusted_prob = 0.5 + (base_win_prob - 0.5) * weather_factor * duration_factor
            win_prob = 0.5 + (base_win_prob - 0.5) * weather_factor * duration_factor
            win_prob = max(0.01, min(0.99, win_prob))  # capping between 1% to 99%
            
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

# Header image - Pickleball match photo
# Option 1: Use a local image file (recommended)
# Place your image file (e.g., pickleball_header.jpg or pickleball_header.png) 
# in the same folder as this script, then uncomment the line below:
# st.image("pickleball_header.jpg", use_container_width=True)

# Option 2: Use an image URL
# Uncomment the line below and replace with your image URL:
# st.image("https://example.com/pickleball-match.jpg", use_container_width=True)

# For now, the image is commented out - uncomment one of the options above to display your image

# Displays the title and instruction for user 
st.title("🥒🏓 Pickleball Winner Predictor")
st.write("Hello pickleball fanatic! Want to predict your next favourite player's match? \nEnter the match conditions and get a prediction!")


# Initialize session state with default values if not already set
if 'p1_rank' not in st.session_state:
    st.session_state.p1_rank = 4.5
if 'p2_rank' not in st.session_state:
    st.session_state.p2_rank = 4.5
if 'p1_win_rate' not in st.session_state:
    st.session_state.p1_win_rate = 0.65
if 'p2_win_rate' not in st.session_state:
    st.session_state.p2_win_rate = 0.65
if 'court_type' not in st.session_state:
    st.session_state.court_type = "indoor"
if 'weather' not in st.session_state:
    st.session_state.weather = "indoor_aircon"
if 'duration' not in st.session_state:
    st.session_state.duration = 60

# Check if we need to randomize (before widgets are created)
if 'should_randomize' in st.session_state and st.session_state.should_randomize:
    st.session_state.p1_rank = round(np.random.uniform(3.0, 6.0), 1)
    st.session_state.p2_rank = round(np.random.uniform(3.0, 6.0), 1)
    st.session_state.p1_win_rate = round(np.random.uniform(0.4, 0.9), 2)
    st.session_state.p2_win_rate = round(np.random.uniform(0.4, 0.9), 2)
    st.session_state.court_type = np.random.choice(["indoor", "outdoor"])
    if st.session_state.court_type == "indoor":
        st.session_state.weather = "indoor_aircon"
    else:
        st.session_state.weather = np.random.choice(["sunny", "windy", "cloudy"])
    st.session_state.duration = np.random.randint(45, 75)
    st.session_state.should_randomize = False  # Reset flag

st.subheader("Match conditions")

# Two-column layour for player inputs
col1, col2 = st.columns(2)

# Left column for player 1 stats
with col1:
    p1_rank = st.slider("Player 1 Rank", 3.0, 6.0, value=st.session_state.p1_rank, step=0.1, key="p1_rank")
    p1_win_rate = st.slider("Player 1 Win Rate", 0.40, 0.90, value=st.session_state.p1_win_rate, step=0.01, key="p1_win_rate")
   
# Right column for player 2 stats
with col2:
    p2_rank = st.slider("Player 2 Rank", 3.0, 6.0, value=st.session_state.p2_rank, step=0.1, key="p2_rank")
    p2_win_rate = st.slider("Player 2 Win Rate", 0.40, 0.90, value=st.session_state.p2_win_rate, step=0.01, key="p2_win_rate")

court_type = st.selectbox("Court type", ["indoor", "outdoor"], index=0 if st.session_state.court_type == "indoor" else 1, key="court_type")

# Condititional weather logic 
if court_type == "indoor":
    weather = "indoor_aircon"
    st.session_state.weather = "indoor_aircon"
    st.info("Indoor court selected. Thus, weather is fixed to air-conditioned indoor.")
else:
    weather_options = ["sunny", "windy", "cloudy"]
    weather_index = weather_options.index(st.session_state.weather) if st.session_state.weather in weather_options else 0
    weather = st.selectbox("Weather", weather_options, index=weather_index, key="weather")

# Match duration user input
duration = st.slider("Duration (minutes)", 45, 75, value=st.session_state.duration, step=1, key="duration")

# Randomize button - placed after inputs for better UX
st.write("Don't know what to input? Just randomise it!")
if st.button("🎲 Randomize inputs", key="randomize_btn"):
    st.session_state.should_randomize = True
    st.rerun()  # Force a rerun to update the widgets
    
# Build input row as DataFrame (must match training column names)
# These inputs are what's get passed to the model/pipeline for prediction
X_input = pd.DataFrame([{
    "p1_rank": p1_rank,
    "p2_rank": p2_rank,
    "p1_win_rate": p1_win_rate,
    "p2_win_rate": p2_win_rate,
    "court_type": court_type,
    "weather": weather,
    "duration": duration
}])

# Runs prediction ONLY when user clicks button 
st.write("Now let's predict the winner:")
if st.button("🏅 Predict winner"):
    pred = model.predict(X_input)[0]        # [0] because we only have one row in the input DataFrame
    proba = model.predict_proba(X_input)[0]

    # Identify prob for predicted class safely
    classes = list(model.named_steps["logisticregression"].classes_)  # Get the model's class order!
    pred_idx = classes.index(pred)  # finds the correct index in the probability array
    confidence = proba[pred_idx]    # extract the confidence
    
    # Get probabilities for both players
    p1_prob = proba[classes.index('Player 1')] if 'Player 1' in classes else 0
    p2_prob = proba[classes.index('Player 2')] if 'Player 2' in classes else 0
    
    # Display results with better UI
    st.markdown("---")
    st.markdown("## 🏆 Prediction Results")
    
    # Winner announcement with conditional styling
    if confidence > 0.7:
        st.success(f"### 🎉 Predicted Winner: **{pred}**")
    elif confidence > 0.55:
        st.warning(f"### ⚖️ Predicted Winner: **{pred}** (Close Match!)")
    else:
        st.info(f"### 🤔 Predicted Winner: **{pred}** (Very Close!)")
    
    # Confidence display with progress bar
    st.markdown(f"**Confidence Level:** {confidence:.2%}")
    st.progress(confidence)
    
    # Probability comparison in columns
    st.markdown("### 📊 Win Probabilities")
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.markdown(f"#### 👤 Player 1")
        st.metric("Win Probability", f"{p1_prob:.2f}")
        st.progress(p1_prob)
    
    with prob_col2:
        st.markdown(f"#### 👤 Player 2")
        st.metric("Win Probability", f"{p2_prob:.2f}")
        st.progress(p2_prob)
    
    # Detailed probability breakdown in a nice format
    st.markdown("### 📈 Detailed Probability Breakdown")
    prob_df = pd.DataFrame({
        'Player': ['Player 1', 'Player 2'],
        'Win Probability': [f"{p1_prob:.2f}", f"{p2_prob:.2f}"]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

# Visual separator 
st.divider()

# Visual Comparison of Predicted vs Actual Outcomes
st.markdown("## 📊 Model Performance: Predicted vs Actual Outcomes")
st.write("Compare model predictions against the original simulated dataset")

if st.button("🔄 Generate Comparison", type="secondary"):
    with st.spinner("Regenerating original dataset and making predictions..."):
        # Regenerate the original dataset
        original_df = generate_pickleball_data(num_matches=200)
        
        # Separate features and actual winners
        X_original = original_df.drop('winner', axis=1)
        y_actual = original_df['winner']
        
        # Make predictions using the trained model
        y_pred = model.predict(X_original)
        y_pred_proba = model.predict_proba(X_original)
        
        # Calculate metrics
        accuracy = accuracy_score(y_actual, y_pred)
        cm = confusion_matrix(y_actual, y_pred, labels=['Player 1', 'Player 2'])
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        with col2:
            correct = sum(y_actual == y_pred)
            st.metric("Correct Predictions", f"{correct}/{len(original_df)}")
        with col3:
            incorrect = sum(y_actual != y_pred)
            st.metric("Incorrect Predictions", f"{incorrect}/{len(original_df)}")
        
        # Confusion Matrix
        st.markdown("### 🔍 Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=['Actual: Player 1', 'Actual: Player 2'],
            columns=['Predicted: Player 1', 'Predicted: Player 2']
        )
        st.dataframe(cm_df, use_container_width=True)
        
        # Visual representation of confusion matrix
        st.markdown("#### Confusion Matrix Visualization")
        # Create a simple bar chart representation
        confusion_data = {
            'True Positive (P1)': cm[0][0],
            'False Negative (P1→P2)': cm[0][1],
            'False Positive (P2→P1)': cm[1][0],
            'True Negative (P2)': cm[1][1]
        }
        confusion_df = pd.DataFrame(list(confusion_data.items()), columns=['Category', 'Count'])
        st.bar_chart(confusion_df.set_index('Category'))
        
        # Classification Report
        st.markdown("### 📈 Detailed Classification Report")
        report = classification_report(y_actual, y_pred, target_names=['Player 1', 'Player 2'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Comparison Table
        st.markdown("### 📋 Sample Match Predictions (First 20 matches)")
        
        # Create comparison dataframe
        comparison_df = original_df.copy()
        comparison_df['predicted_winner'] = y_pred
        comparison_df['prediction_correct'] = comparison_df['winner'] == comparison_df['predicted_winner']
        
        # Add confidence scores
        classes = list(model.named_steps["logisticregression"].classes_)
        confidences = []
        for i, pred in enumerate(y_pred):
            pred_idx = classes.index(pred)
            confidences.append(y_pred_proba[i][pred_idx])
        comparison_df['confidence'] = [f"{c:.1%}" for c in confidences]
        
        # Display first 20 matches
        display_cols = ['p1_rank', 'p2_rank', 'p1_win_rate', 'p2_win_rate', 
                       'court_type', 'weather', 'winner', 'predicted_winner', 
                       'confidence', 'prediction_correct']
        display_df = comparison_df[display_cols].head(20)
        display_df.columns = ['P1 Rank', 'P2 Rank', 'P1 Win Rate', 'P2 Win Rate',
                             'Court', 'Weather', 'Actual Winner', 'Predicted Winner',
                             'Confidence', 'Correct?']
        
        # Color code the dataframe
        def highlight_correct(row):
            if row['Correct?']:
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_correct, axis=1),
            use_container_width=True
        )
        
        # Summary statistics by prediction correctness
        st.markdown("### 📊 Prediction Analysis")
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("#### Accuracy by Actual Winner")
            accuracy_by_winner = comparison_df.groupby('winner')['prediction_correct'].mean()
            st.bar_chart(accuracy_by_winner * 100)
        
        with col5:
            st.markdown("#### Prediction Distribution")
            pred_dist = pd.Series(y_pred).value_counts()
            st.bar_chart(pred_dist)

st.divider()
st.caption("Hope you enjoyed! Feel free to share your thoughts and feedback.")