import streamlit as st
import pickle
import pandas as pd

# List of IPL teams
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals', 'Lucknow Super Giants', 'Gujarat Titans']

# List of cities where IPL matches are held
cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur', 'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Bengaluru', 'Indore', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow', 'Guwahati']

# Load the trained model
pipe = pickle.load(open('pipe.pkl','rb'))

# Streamlit app title
st.title('IPL Win Predictor')

# Splitting the page into two columns for better UI
col1, col2 = st.columns(2)

# Dropdown to select batting team
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

# Dropdown to select bowling team
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Dropdown to select host city
selected_city = st.selectbox('Select host city', sorted(cities))

# Input field for target score
target = st.number_input('Target')

# Splitting the page into three columns
col3, col4, col5 = st.columns(3)

# Input field for current score
with col3:
    score = st.number_input('Score')

# Input field for overs completed
with col4:
    overs = st.number_input('Overs completed')

# Input field for wickets out
with col5:
    wickets = st.number_input('Wickets out')

# Button to trigger prediction
if st.button('Predict Probability'):
    # Calculate remaining runs, balls, wickets, current run rate, and required run rate
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    # Create a DataFrame with input data
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    
    # Predict probabilities using the model
    result = pipe.predict_proba(input_df)
    
    # Extract win and loss probabilities
    loss = result[0][0]
    win = result[0][1]
    
    # Display win and loss probabilities
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
