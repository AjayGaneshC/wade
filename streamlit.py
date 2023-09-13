import streamlit as st
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define your custom CSS styles
custom_css = """
<style>
body {
    background-color: #800080; /* Purple background color */
    font-family: Arial, sans-serif;
    color: white;
}
h1 {
    text-align: center;
    color: #ff5722; /* Orange title color */
    font-size: 2.5em;
    padding: 20px;
    text-transform: uppercase;
}
h2 {
    color: #ff5722; /* Orange header color */
    font-size: 1.5em;
    margin-top: 20px;
    margin-bottom: 10px;
}
.container {
    padding: 20px;
}
</style>
"""

# Apply custom CSS styles
st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.markdown("<h1>VIZAI - Regression Visualization</h1>", unsafe_allow_html=True)

# Container for content
container = st.container()

# Sidebar
with container:
    st.markdown("<h2>Regression Options</h2>")

    # Add an option for users to upload their dataset
    uploaded_file = st.file_uploader("Upload your own dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        custom_df = pd.read_csv(uploaded_file)

        # Check if the dataset has multiple columns (features)
        if len(custom_df.columns) < 2:
            st.error("The uploaded dataset should have at least two columns (features).")
        else:
            # Separate features and target variable
            X = custom_df.iloc[:, :-1].values  # Features
            y = custom_df.iloc[:, -1].values  # Target variable

            # Select regression algorithm
            algorithm = st.selectbox(
                'Select Algorithm',
                ('Logistic Regression', 'Linear Regression', 'Polynomial Regression')
            )

            penalty = st.selectbox(
                'Regularization',
                ('l2', 'l1', 'elasticnet', 'none')
            )

            c_input = st.number_input('C', value=1.0)

            solver = st.selectbox(
                'Solver',
                ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
            )

            max_iter = st.number_input('Max Iterations', value=100)

            multi_class = st.selectbox(
                'Multi Class',
                ('auto', 'ovr', 'multinomial')
            )

            l1_ratio = st.number_input('l1 Ratio')

            if algorithm == 'Polynomial Regression':
                degree = st.slider('Polynomial Degree', min_value=2, max_value=10, value=2)

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

            # Apply the selected algorithm and plot
            if algorithm == 'Logistic Regression':
                clf = LogisticRegression(penalty=penalty, C=c_input, solver=solver, max_iter=max_iter,
                                        multi_class=multi_class, l1_ratio=l1_ratio)
                clf.fit(X_train, y_train)  # Fit the model before making predictions

                y_pred = clf.predict(X_test)

                # Plot decision boundary
                # (You may need to modify the plotting logic depending on your dataset)

            elif algorithm == 'Linear Regression':
                clf = LinearRegression()
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                # Plot the regression line and data points
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.scatter(X_train, y_train, color='#009688', s=50)
                plt.plot(X_train, clf.predict(X_train), color='#ff5722', linewidth=2)
                plt.xlabel("Feature")
                plt.ylabel("Target Variable")
                plt.title("Linear Regression")
                st.pyplot(plt)

            elif algorithm == 'Polynomial Regression':
                poly_features = PolynomialFeatures(degree=degree)
                X_poly = poly_features.fit_transform(X_train)
                clf = LinearRegression()
                clf.fit(X_poly, y_train)

                y_pred = clf.predict(poly_features.transform(X_test))

                # Plot the regression curve and data points
                plt.figure(figsize=(8, 6))
                # Modify this line to specify the columns for X and y
                plt.scatter(X, y, color='#009688', s=50)

                X_poly_plot = poly_features.transform(X)
                plt.plot(X, clf.predict(X_poly_plot), color='#ff5722', linewidth=2)
                plt.xlabel("Feature")
                plt.ylabel("Target Variable")
                plt.title("Polynomial Regression (Degree " + str(degree) + ")")
                st.pyplot(plt)

# Load initial graph
# Here, you can load a default graph or provide an option to upload another dataset as needed.

# Run the Streamlit app
if __name__ == '__main__':
    container.markdown("---")
    container.markdown("© 2023 VIZAI. All rights reserved----AJAY C AKA WADE.")
