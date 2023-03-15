import matplotlib.pyplot as plt
import seaborn as sns     
import numpy as np
import streamlit as st
import pickle
import pandas as pd
import webbrowser

from PIL import Image

pickle_in = open("rfr2_model.pkl","rb")
model=pickle.load(pickle_in)


def predict_intrusion(Plant,Customer):
    
    prediction=model.predict([[Plant,Customer]])
    print(prediction)
    return prediction


# Set option to suppress deprecation warnings for pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load your dataset
df = pd.read_csv('shipment.csv')

# Define function to plot histogram
def plot_histogram(col):
    plt.hist(df[col])
    plt.xlabel(col)
    plt.ylabel('Frequency')
    st.pyplot()

# Define function to plot scatterplot
def plot_scatter(col1, col2):
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    st.pyplot()

# Define function to plot heatmap
def plot_heatmap():
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    st.pyplot()

# Define function to plot pie chart
def plot_pie(col, key):
    counts = df[col].value_counts()
    plt.pie(counts, labels=counts.index.tolist(), autopct='%1.1f%%')
    plt.title(col)
    st.pyplot(key=key)
    

# Define function to plot line chart
def plot_line(col1, key):
    plt.plot(df[col1])
    plt.xlabel(col1)
    plt.ylabel('Frequency')
    st.pyplot(key=key)

    





# Define Streamlit app
  
def main():
   
    html_temp = """
    <div style="background-colour:black;padding:15px">
    <h1 style="color:white;text-align:center;">Forecasting Movement</h1>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    
    Plant = st.text_input("Plant",placeholder="Range:0 to 18")
    
    Customer = st.text_input("Customer",placeholder="Range:0 to 205658")
    
    result=""
    if st.button("Predict"):
        result=predict_intrusion(Plant,Customer)
    st.success('The Movement is {}'.format(result))
    
    st.title('Exploratory Data Analysis')
    # Define the external link URL
    external_link = 'https://genga28.github.io/Buckman/'
    # Create the button
    if st.button('View profiling report'):
        webbrowser.open_new_tab(external_link)

    # Display dataset
    st.subheader('Dataset')
    st.write(df.head())

    # Display column selection for histogram
    st.subheader('Histogram')
    col1 = st.selectbox('Select a column', df.columns, key='hist_col_select')
    plot_histogram(col1)
    

    # Display column selection for scatterplot
    st.subheader('Scatterplot')
    col2 = st.selectbox('Select first column', df.columns, key='scatter_col1_select')
    col3 = st.selectbox('Select second column', df.columns, key='scatter_col2_select')
    plot_scatter(col2, col3)

    # Display pie chart
    st.subheader('Pie Chart')
    col4 = st.selectbox('Select a column', df.columns, key='pie_col_select')
    plot_pie(col4, key='pie_chart')

    # Display line chart
    st.subheader('Line Chart')
    col5 = st.selectbox('Select first column', df.columns, key='line_col1_select')
    
    plot_line(col5, key='line_chart')

    # Display heatmap
    st.subheader('Correlation heatmap')
    plot_heatmap()

    

if __name__ == "__main__":
    main()