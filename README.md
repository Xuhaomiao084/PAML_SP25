# PAML SPRING 2025 Final Project - Airbnb Price Predictor

This project implements a machine learning model to predict Airbnb prices in New York City based on various features. The project consists of three main components: data analysis, model implementation, and a user interface.

## Project Structure

```
.
├── data/               # Dataset directory containing Airbnb data
├── new_project.ipynb   # Jupyter notebook with data analysis and model implementation
├── app.py             # Streamlit web application for price prediction
└── requirements.txt   # Python dependencies
```

## Features

The price prediction model takes into account the following features:
- Room Type (Entire home/apt, Private room, Shared room)
- Neighborhood (NYC boroughs)
- Number of Nights
- Number of Reviews per Month

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Analysis and Model Training
To view the data analysis and model training process:
1. Open `new_project.ipynb` in Jupyter Notebook or Jupyter Lab
2. Run the cells to see the analysis and train the model

### Price Prediction Interface
To use the price prediction interface:
1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Input the required information:
   - Select the room type
   - Choose the neighborhood
   - Enter the number of nights
   - Specify the number of reviews per month
3. Click "Predict Price" to get the estimated price

## Dependencies

The main dependencies for this project are:
- pandas
- numpy
- streamlit
- scikit-learn
- matplotlib
- seaborn

All dependencies are listed in `requirements.txt`

## Data

The dataset used in this project is located in the `data/` directory and contains information about Airbnb listings in New York City, including features such as room type, neighborhood, price, and review statistics.

## Model

The price prediction model uses linear regression with gradient descent optimization. The model is trained on historical Airbnb data to learn the relationships between various features and listing prices.

## Contributing

Feel free to submit issues and enhancement requests! 
