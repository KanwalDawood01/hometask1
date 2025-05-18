import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.title("Simple Linear Regression Using NumPy")
st.write("Upload a CSV file with 'Duration' and 'Calories' columns")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Check columns
    if 'Duration' not in df.columns or 'Calories' not in df.columns:
        st.error("CSV must contain 'Duration' and 'Calories' columns")
    else:
        # Extract features and targets
        X = df['Duration'].values.reshape(-1, 1)
        y = df['Calories'].values.reshape(-1, 1)

        # Split into training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize parameters: slope and intercept
        w = np.random.randn()
        b = np.random.randn()
        lr = 0.0001
        epochs = 1000

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            y_pred = w * X_train + b
            train_loss = np.mean((y_pred - y_train) ** 2)
            train_losses.append(train_loss)

            # Gradient calculation
            dw = np.mean(2 * (y_pred - y_train) * X_train)
            db = np.mean(2 * (y_pred - y_train))

            # Parameter update
            w -= lr * dw
            b -= lr * db

            # Validation loss
            y_val_pred = w * X_val + b
            val_loss = np.mean((y_val_pred - y_val) ** 2)
            val_losses.append(val_loss)

        st.subheader("Final Model Parameters")
        st.write(f"Slope (w): {w:.4f}")
        st.write(f"Intercept (b): {b:.4f}")

        # Plot: Regression Line
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y, color='blue', label='Data Points')
        ax1.plot(X, w * X + b, color='red', label='Regression Line')
        ax1.set_xlabel("Duration")
        ax1.set_ylabel("Calories")
        ax1.legend()
        st.subheader("Regression Line")
        st.pyplot(fig1)

        # Plot: Loss curves
        fig2, ax2 = plt.subplots()
        ax2.plot(train_losses, label='Train Loss')
        ax2.plot(val_losses, label='Validation Loss')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss")
        ax2.legend()
        st.subheader("Loss per Epoch")
        st.pyplot(fig2)
