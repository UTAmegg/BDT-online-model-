import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Test App")
st.write("If you see this, matplotlib is working!")

# Simple test plot
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))
st.pyplot(fig)