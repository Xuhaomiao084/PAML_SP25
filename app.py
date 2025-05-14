import streamlit as st
import numpy as np
import pandas as pd

def gradient_descent_linear(X, y, learning_rate=0.01, n_iterations=1000):
    # Add a column of ones to X for the bias term
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    # Initialize parameters (weights) to zeros
    theta = np.zeros(X_b.shape[1])
    
    # Lists to store cost history
    cost_history = []
    
    m = len(y)  # number of training examples
    
    for i in range(n_iterations):
        # Forward pass (predictions)
        y_pred = X_b.dot(theta)
        
        # Compute gradients
        gradients = 2/m * X_b.T.dot(y_pred - y)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Compute cost
        cost = np.mean((y_pred - y) ** 2)
        cost_history.append(cost)
    
    return theta, cost_history

def predict_price(room_type, neighborhood, nights, reviews_per_month, model_params):
    # Create feature vector
    features = np.zeros(len(model_params) - 1)  # -1 for the bias term
    
    # Set the corresponding indices for room_type and neighborhood
    # These would normally be determined by your training data encoding
    # For now, we'll use simple numerical encoding
    features[0] = {'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2}[room_type]
    features[1] = hash(neighborhood) % 10  # Simple hash function for demonstration
    features[2] = nights
    features[3] = reviews_per_month
    
    # Add bias term
    X = np.concatenate(([1], features))
    
    # Make prediction
    price = X.dot(model_params)
    return max(0, price)  # Ensure non-negative price

def main():
    st.title("Airbnb Price Predictor")
    st.write("Enter the details below to get a price prediction")

    # Input fields
    room_types = ['Entire home/apt', 'Private room', 'Shared room']
    neighborhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']  # Example neighborhoods
    
    # Create columns for a better layout
    col1, col2 = st.columns(2)
    
    with col1:
        room_type = st.selectbox("Room Type", room_types)
        neighborhood = st.selectbox("Neighborhood", neighborhoods)
    
    with col2:
        nights = st.number_input("Number of Nights", min_value=1, max_value=365, value=1)
        reviews = st.number_input("Number of Reviews per Month", min_value=0, max_value=100, value=0, step=1)

    # For demonstration, we'll use some dummy model parameters
    # In a real application, these would come from your trained model
    dummy_model_params = np.array([100, -20, 10, 5, 2])  # Bias, room_type, neighborhood, nights, reviews

    if st.button("Predict Price"):
        price = predict_price(room_type, neighborhood, nights, reviews, dummy_model_params)
        st.success(f"Predicted Price: ${price:.2f} per night")
        
        # Add some context
        st.info("""
        This prediction is based on:
        - Room Type: {room_type}
        - Neighborhood: {neighborhood}
        - Stay Duration: {nights} nights
        - Reviews per Month: {reviews}
        """)

if __name__ == "__main__":
    main() 