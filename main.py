import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Set up the page configuration
st.set_page_config(layout="wide")
st.title("Are you paying a fair Rent in Berlin?")

# Sample historical data for district and year-wise price per m² (for demonstration)
historical_data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Price Per m2 (42.5m2)': [12.87, 14.57, 14.57, 14.55, 15.76, 16.61, 17.12, 18.91, 20.92, 21.99, 21.60, 25.17, 30.05, 30.99, 29.64]
}

# Convert to DataFrame
df = pd.DataFrame(historical_data)

# Fit the linear regression model
model = LinearRegression()
X = df['Year'].values.reshape(-1, 1)
y = df['Price Per m2 (42.5m2)']
model.fit(X, y)

# Calculate MAE on historical data
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)

# Fair Rent Check Section
col1, col2, col3 = st.columns((0.4, 0.1, 0.5))

with col1:
    st.header("Fair Rent Check")
    # Rent and square meter input
    rent_amount = st.number_input("Enter the Rent amount you are paying (in Euros):", min_value=0)
    square_meter = st.number_input("Enter the area in square meters:", min_value=1.0)

    # Year input
    year = st.selectbox("Select Year:", options=['2018', '2019', '2020', '2021', '2022', '2023', '2024'], index=6)

    # When user clicks Check, calculate the results
    if st.button("Check"):
        # Predict price per m² for the selected year
        year_int = int(year)
        predicted_price_per_m2 = model.predict(np.array([[year_int]]))[0]

        # Calculate the fair rent for the entered square meter area
        fair_rent = predicted_price_per_m2 * square_meter

        # Compare with user input rent and display results
        difference = rent_amount - fair_rent
        if difference > 0:
            st.write(f"You're paying €{difference:.2f} more than the fair rent for {square_meter} m².")
        elif difference < 0:
            st.write(f"You're paying €{-difference:.2f} less than the fair rent for {square_meter} m².")
        else:
            st.write("You're paying the fair rent for the given area.")

        st.write(f"Fair rent for {square_meter} m² in {year_int}: €{fair_rent:.2f}")
        st.write(f"Predicted price per m² in {year_int} for 1 bedroom apartment (42m2 on average): €{predicted_price_per_m2:.2f}")

with col3:
    st.header("Future Rent Price Predictions")

    # Select a future year from the dropdown
    selected_year = st.selectbox("Select a year for future rent price prediction:", options=[2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034])

    # Predict rent for the selected year
    selected_year_array = np.array([[selected_year]])
    predicted_rent_per_m2 = model.predict(selected_year_array)[0]
    optimistic = predicted_rent_per_m2 + mae
    pessimistic = predicted_rent_per_m2 - mae

    # Show the predicted range of rent per m² for the selected year
    st.markdown(f"##### Predicted Rent per m² for {selected_year}: €{predicted_rent_per_m2:.2f} OR between €{pessimistic:.2f} to €{optimistic:.2f}")

    # Predict and plot future years with ranges
    future_years = np.array([2025, 2026, 2027, 2028]).reshape(-1, 1)
    future_predictions = model.predict(future_years)
    optimistic_predictions = future_predictions + mae
    pessimistic_predictions = future_predictions - mae

    # Plot the historical and future predictions
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['Price Per m2 (42.5m2)'], label="Historical Data", marker='o', linestyle='-', color='gray')
    plt.plot(np.append(df['Year'], future_years.flatten()), np.append(df['Price Per m2 (42.5m2)'], future_predictions), '--', color='blue', alpha=0.7, label="Predicted Prices")
    plt.fill_between(
        np.append(df['Year'], future_years.flatten()), 
        np.append(df['Price Per m2 (42.5m2)'], pessimistic_predictions),
        np.append(df['Price Per m2 (42.5m2)'], optimistic_predictions),
        color='blue', alpha=0.2, label="Prediction Range"
    )

    plt.title("Forecasted Rent per m² with Ranges")
    plt.xlabel('Year')
    plt.ylabel('Price per m² (€)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    st.pyplot(plt)
