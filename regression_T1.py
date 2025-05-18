import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("NumPy Linear Regression App")

# Data generation
true_w = 2
true_b = 1
N = 100
np.random.seed(100)
x = np.random.rand(N, 1)
epsilon = 0.1 * np.random.randn(N, 1)
y = true_w * x + true_b + epsilon

# Split into train and validation
idx = np.arange(N)
np.random.shuffle(idx)
idx_train = idx[:int(0.8 * N)]
idx_test = idx[int(0.8 * N):]
x_train, y_train = x[idx_train], y[idx_train]
x_val, y_val = x[idx_test], y[idx_test]

st.subheader("Initial Data")
fig1, ax1 = plt.subplots()
ax1.scatter(x_train, y_train, label="Train")
ax1.scatter(x_val, y_val, label="Validation", color='m')
ax1.set_title(f"True w = {true_w}, True b = {true_b}")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
st.pyplot(fig1)

if st.button("Train Model"):
    # Training loop
    trainLosses = []
    valLosses = []
    lr = 0.1
    w = np.random.randn(1)
    b = np.random.randn(1)

    for i in range(100):
        yhat = w * x_train + b
        error = yhat - y_train
        loss = (error ** 2).mean()
        trainLosses.append(loss)

        db = 2 * error.mean()
        dw = 2 * (x_train * error).mean()
        b -= lr * db
        w -= lr * dw

        yhatVal = w * x_val + b
        errorVal = yhatVal - y_val
        valLoss = (errorVal ** 2).mean()
        valLosses.append(valLoss)

        if valLoss < 0.0001:
            break

    st.success(f"Training complete at epoch {i+1}")
    st.write(f"Final w: {w[0]:.4f}, Final b: {b[0]:.4f}")
    st.write(f"Train Loss: {loss:.6f}, Val Loss: {valLoss:.6f}")

    # Final regression line plot
    fig2, ax2 = plt.subplots()
    ax2.scatter(x_train, y_train, label="Train")
    ax2.scatter(x_train, yhat, label="Train Predictions")
    ax2.plot(x, w * x + b, color='r', label='Regression Line')
    ax2.set_title("Training Data and Regression Line")
    ax2.legend()
    st.pyplot(fig2)

    # Validation predictions
    fig3, ax3 = plt.subplots()
    ax3.scatter(x_val, y_val, label="Validation")
    ax3.scatter(x_val, yhatVal, label="Validation Predictions")
    ax3.plot(x, w * x + b, color='orange', label='Regression Line')
    ax3.set_title("Validation Data and Predictions")
    ax3.legend()
    st.pyplot(fig3)

    # Train Loss vs Epoch
    fig4, ax4 = plt.subplots()
    ax4.plot(trainLosses, label="Train Loss")
    ax4.set_title("Train Loss vs Epoch")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MSE")
    st.pyplot(fig4)

    # Validation Loss vs Epoch
    fig5, ax5 = plt.subplots()
    ax5.plot(valLosses, label="Validation Loss", color='m')
    ax5.set_title("Validation Loss vs Epoch")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("MSE")
    st.pyplot(fig5)
