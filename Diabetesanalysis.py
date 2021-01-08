# Importing Libraries
# for some basic operations
import streamlit as st
import numpy as np
import pandas as pd
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import plotting
from pandas.plotting import parallel_coordinates
# for interactive visualizations
import plotly
import plotly.offline as py
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
#for prediction
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

st.write ("""<style> body {
    background-color: #E3E3E3;
} </style>""", unsafe_allow_html=True)
    #############################################################

#Importing the dataset for visualizations
data1=pd.read_csv(r'C:\Users\jana\Desktop\Diabetes\diabetescount.csv')
data2=pd.read_csv(r'C:\Users\jana\Desktop\Diabetes\Earlydiabetes.csv')
#Importing the dataset for prediction
data3=pd.read_csv(r'C:\Users\jana\Desktop\Diabetes\diabetes.csv')
                ##############################################################

hide_streamlit= """  <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;}  </style> """
st.markdown(hide_streamlit, unsafe_allow_html=True)
                ####################################################################

# To put a picture in the subheader
from PIL import Image
image = Image.open('Diabestes.png')
st.image(image,width=None)

                ####################################################################

# Adding background
st.write ("""<style> body {
background-image: url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUSEhIVFRUXFxcVFRUVFRUVFRUVFxcXFxUVFRUYHSggGBolHRUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKBQUFDgUPDisZExkrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrKysrK//AABEIAOEA4QMBIgACEQEDEQH/xAAXAAEBAQEAAAAAAAAAAAAAAAABAAIH/8QAIhABAQEBAQEAAQMFAAAAAAAAAAERAjFBUSHB8BJhgZGh/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAH/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwDstUKkUOM9NsAT0CAoNWAIaOTQV9BvqBKlkCe1FaAiio0DFVyQSWCAaGtZoKGIAbAbWYDSQAwQiAtZa+gDGmToCoIFDYuT0AHJxQFEKYC01RUBRTF0ChCBrlnTBgKJVAI0MMAWMtLASOIEDBoFdKgDyhDAFiiq5oKLocGgRp1mAYoogVLNIJVaIDQMZoNyAyjQZ6PIqgGKEAWTQDWofqgMC04Co5N8UBKRKUAuSALPTcZoCFVUFUqbAZP4EagANdMwGoyRQMM9FX0FV+EYCEQgKlVSAciOoGOWoJDAFXKMgJAyAglAIw6tBM30rAMNZaoMQ8xKgaKYKBjNaQBRVQCDV19BlQ9LkDf3ZjVWAEUAiUMANQGUADGQa5ODkAiCAULMApVaBkBiwAkQStUFAwRKAaqFYBsSsVAT1L6oBRQAys1QGhyviAaqqqC5Vii0FwRydBT4DFQUGI0FENOgKrVUCiKgM0xVAQkBHR1m0DEQB1LEAq4VUA4pUKBCMAClAGrAdBc1USmgDVEDOGj6QQpqApABCKYCwqKeAKFSB1CID+n5TP8AhATUyDTNaAKIQwF8U+oA3WSqDLTJA2iKqAl9VMABU4BqFICnRFAIMVBm1b+qqgNT6J4uTPKDOBIGrVEQFBQCGCQghTVQUiMHXgLlHkAqWWgZrXLNrUoDpEAmsZqAhaKDUqZrQM4jVAM/ZfkVAP6f7ppAzWmaYBgwwUEolyB6rP1qwYBZ6rVZsBrgGAFCLEAhiwwEFECUWGABTRQVNGGgbGZDaQCUioFMoEeYlASVUANSAgaFRAQOoDzFfVyqCoIoI8s41AQSoL6YyaAWFUFpsZjUAVQVQGxUACaxAy1RIQHwHpSADEgOiDSANUNoCU0VApVUL6DUViotA1CEB9IhkBCmKAMMgMoM0wUgay0zAOpagPINAG+AgFDGYQSWmAII1BQRZIKrCzQaoMqBM2tDAEagiA4COQS0U0BSrEBrMaAJFAumakBgqQGGJAoIkDXLNSBQ/QgLNCBqmJAuf5/tT2pAgEBvkXPqQK+m/wA/6kAhiQJVIG0kD//Z");
background-repeat: repeat; } </style>""", unsafe_allow_html=True)

st.write(""" <style>body {
              margin: 40px;}
            .box {
              background-color: #444;
              color: #fff;
              border-radius: 5px;
              padding: 20px;
              font-size: 150%;}
            .box:nth-child(even) {
              background-color: #ccc;
              color: #000;}
            .wrapper {
              width: 600px;
              display: grid;
              grid-gap: 10px;
              grid-template-columns: repeat(6, 100px);
              grid-template-rows: 100px 100px 100px;
              grid-auto-flow: column;}</style>""",unsafe_allow_html=True)
                ####################################################################

#Title and header information
st.markdown("<h2 style='text-align: center; color: firebrick;'>Diabetes and Early Stages </h2>", unsafe_allow_html=True)
st.text(" \n")
st.markdown("""<h3 style='text-align: justify; color: grey;'> Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.</h3>""", unsafe_allow_html=True)
st.markdown("""<h3 style='text-align: justify; color: grey;'> How can you tell if you have diabetes? Most early symptoms are from higher-than-normal levels of glucose, a kind of sugar, in your blood. A study using several Diabetes Data set from Kaggle was done to demonstrate early diabetes stages in addition to a predictive model that predicts if a person is healthy or not.</h3>""", unsafe_allow_html=True)
st.text(" \n")
                ####################################################################

#Starting to build the dashboard
st.markdown("""<h3><span style="color:firebrick">Areas Addressed</h3>""", unsafe_allow_html=True)
                ####################################################################

#Building the side bar
st.sidebar.title("Diabetes and Early Stages:")
st.sidebar.markdown("""<p><b>Data Source: <a href="https://www.kaggle.com/sujan97/early-stage-diabetes-2020" target="_blank">Early Stage Diabetes</a></p></div></div></div>""",unsafe_allow_html=True)
st.text(" \n")
st.sidebar.markdown("""<h3><span style="color:firebrick">Areas Addressed:</h3>""",unsafe_allow_html=True)
if st.sidebar.checkbox("Tackled Areas"):
    if st.checkbox('Areas Addressed and Analyzed'):
        st.markdown("""The below health areas will be tackled throughout the analysis and dashboard in order to identify how diabetes can be detected and prevented:
            \n 1.Diabetes Distribution across Gender and Age
            \n 2.Diabetes Correlation with Different Attributes
            \n 3.Diabetes Prediction""")
                ####################################################################

st.markdown("""<h3><span style="color:firebrick">Diabetes Analysis and Data Exploration</h3>""", unsafe_allow_html=True)
st.sidebar.markdown("""<h3><span style="color:firebrick">Diabetes Analysis and Data Exploration:</h3>""",unsafe_allow_html=True)
if st.sidebar.checkbox("A Sample of the Data Used"):
    if st.checkbox('First Couple of Rows'):
        st.write(data2.head(10))

df=data2.loc[(data2['class']=='Positive')]
if st.sidebar.checkbox("Select the Desired Analysis of Attribute to Diabetes "):
    attributes_filtered=[]
    option = st.sidebar.selectbox(' ', ('By Country','By Gender','By Age'))
    attributes_filtered.append(['By Country','By Gender','By Age'])
    if (option=="By Country"):
        st.sidebar.markdown('You selected Analysis: _By Country_')
        mapdata=pd.read_csv(r'C:\Users\jana\Desktop\Diabetes\diabetescount.csv')
        import plotly.express as px
        map = px.scatter_mapbox(mapdata, lat="Lat", lon="Lon",
        size='Value',color_discrete_sequence=['firebrick'], zoom=1, height=500)
        map.update_layout(mapbox_style="carto-positron")
        map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(map)


    if (option=="By Gender"):
        st.sidebar.markdown('You selected Analysis: _By Gender_')
        def Gender_window():
            layout = go.Layout(
                autosize=False,
                width=900,
                height=500,
                title='Gender Distribution of Diabetic People',margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=50,
                pad = 4
                ))
            labels = df['Gender'].value_counts().index
            values = df['Gender'].value_counts().values
            fig = {"data": [
                            {
                            "values": values,
                            "labels": labels,
                            "domain": {"x": [0, .48]},
                            "name": "Criticality",
                            "sort": False,
                            "marker": {'colors':["grey","firebrick"]},
                            "textinfo":"percent",
                            "textfont": {'color': '#FFFFFF', 'size': 15},
                            "type": "pie"
                            } ],
                            "layout": layout
                            }
            st.plotly_chart(fig)
        Gender_window()

    if (option=="By Age"):
        st.sidebar.markdown('You selected Analysis: _By Age_')
        fig2 = px.histogram(df, x="Age", title='Distribution of Diabetic People Across Age',
                               opacity=0.8,
                               log_y=True, # represent bars with log scale
                               color_discrete_sequence=['firebrick'])
        st.plotly_chart(fig2)

            ############################################################################################################################
st.markdown("""<h3><span style="color:firebrick">Correlation with Attributes and Early Stage Symptoms</h3>""", unsafe_allow_html=True)
st.sidebar.markdown("""<h3><span style="color:firebrick">Correlation with Attributes:</h3>""",unsafe_allow_html=True)
st.sidebar.markdown("""<h4><span style="color:grey">Choose Analysis Type:</h3>""",unsafe_allow_html=True)
st.sidebar.text(" \n")
if st.sidebar.checkbox("Correlation Heat Map"):
    sns.heatmap(data3.corr(),annot=False,linewidths=0,vmax=0.8, square=True,cmap='gray_r')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
if st.sidebar.checkbox("Early Stages Analysis"):
    gender=st.multiselect('You selected Analysis: By Gender',data2.Gender.unique())
    col1,col2,col3=st.beta_columns(3)
    df1=data2.loc[(data2['Gender'].isin(gender))&(data2['class']=='Positive')]

    def weight_window():
        layout = go.Layout(
            autosize=False,
            width=900,
            height=500
            ,title='Sudden Weight Loss',margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=50,
            pad = 4))
        labels = df1['sudden weight loss'].value_counts().index
        values = df1['sudden weight loss'].value_counts().values
        pie1 = {"data": [
                    {
                    "values": values,
                    "labels": labels,
                    "domain": {"x": [0, .48]},
                    "name": "Criticality",
                    "sort": False,
                    "marker": {'colors':["grey","firebrick"]},
                    "textinfo":"percent",
                    "textfont": {'color': '#FFFFFF', 'size': 15},
                    "type": "pie"
                    } ],
                    "layout": layout
                    }
        col1.plotly_chart(pie1)

    weight_window()

    def itchy_window():
        layout = go.Layout(
                autosize=False,
                width=900,
                height=500
                ,title='Itchiness',margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=50,
                pad = 4
                ))
        labels = df1['Itching'].value_counts().index
        values = df1['Itching'].value_counts().values
        pie2 = {"data": [
                    {
                    "values": values,
                    "labels": labels,
                    "domain": {"x": [0, .48]},
                    "name": "Criticality",
                    "sort": False,
                    "marker": {'colors':["grey","firebrick"]},
                    "textinfo":"percent",
                    "textfont": {'color': '#FFFFFF', 'size': 15},
                    "type": "pie"
                    } ],
                    "layout": layout
                    }
        col1.plotly_chart(pie2)

    itchy_window()

    def muscle_window():
        layout = go.Layout(
                autosize=False,
                width=900,
                height=500
                ,title='Muscle Stiffness',margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=50,
                pad = 4
                ))
        labels = df1['muscle stiffness'].value_counts().index
        values = df1['muscle stiffness'].value_counts().values
        pie4 = {"data": [
                    {
                    "values": values,
                    "labels": labels,
                    "domain": {"x": [0, .48]},
                    "name": "Criticality",
                    "sort": False,
                    "marker": {'colors':["grey","firebrick"]},
                    "textinfo":"percent",
                    "textfont": {'color': '#FFFFFF', 'size': 15},
                    "type": "pie"
                    } ],
                    "layout": layout
                    }
        col2.plotly_chart(pie4)

    muscle_window()

    def healing_window():
        layout = go.Layout(
                autosize=False,
                width=900,
                height=500
                ,title='Delayed Healing',margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=50,
                pad = 4
                ))
        labels = df1['delayed healing'].value_counts().index
        values = df1['delayed healing'].value_counts().values
        pie5 = {"data": [
                    {
                    "values": values,
                    "labels": labels,
                    "domain": {"x": [0, .48]},
                    "name": "Criticality",
                    "sort": False,
                    "marker": {'colors':["grey","firebrick"]},
                    "textinfo":"percent",
                    "textfont": {'color': '#FFFFFF', 'size': 15},
                    "type": "pie"
                    } ],
                    "layout": layout
                    }
        col2.plotly_chart(pie5)

    healing_window()

    def visual_window():
        layout = go.Layout(
                autosize=False,
                width=900,
                height=500
                ,title='Visual Blurring',margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=50,
                pad = 4
                ))
        labels = df1['visual blurring'].value_counts().index
        values = df1['visual blurring'].value_counts().values
        pie5 = {"data": [
                    {
                    "values": values,
                    "labels": labels,
                    "domain": {"x": [0, .48]},
                    "name": "Criticality",
                    "sort": False,
                    "marker": {'colors':["grey","firebrick"]},
                    "textinfo":"percent",
                    "textfont": {'color': '#FFFFFF', 'size': 15},
                    "type": "pie"
                    } ],
                    "layout": layout
                    }
        col3.plotly_chart(pie5)

    visual_window()

    def weakness_window():
        layout = go.Layout(
                autosize=False,
                width=900,
                height=500
                ,title='Weakness',margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=50,
                pad = 4
                ))
        labels = df1['weakness'].value_counts().index
        values = df1['weakness'].value_counts().values
        pie6 = {"data": [
                    {
                    "values": values,
                    "labels": labels,
                    "domain": {"x": [0, .48]},
                    "name": "Criticality",
                    "sort": False,
                    "marker": {'colors':["grey","firebrick"]},
                    "textinfo":"percent",
                    "textfont": {'color': '#FFFFFF', 'size': 15},
                    "type": "pie"
                    } ],
                    "layout": layout
                    }
        col3.plotly_chart(pie6)

    weakness_window()

            ##############################################################################################

#Predictive Analysis

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
# Split dataset into training set and test set
X_train, X_test ,y_train, y_test = train_test_split(data3[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']], data3['Outcome'], test_size=0.3, random_state=109)
#Creating the model
logisticRegr = LogisticRegression(C=1)
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
#Saving the Model
pickle_out = open("logisticRegr.pkl", "wb")
pickle.dump(logisticRegr, pickle_out)
pickle_out.close()

pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.sidebar.markdown("""<h3><span style="color:firebrick">Diabetes Prediction:</h3>""",unsafe_allow_html=True)
st.sidebar.markdown("""<p><b>Data Source: <a href="https://www.kaggle.com/uciml/pima-indians-diabetes-database" target="_blank">Diabetes Prediction</a></p></div></div></div>""",unsafe_allow_html=True)
select = st.sidebar.selectbox('Form', ['Prediction Form'], key='1')
st.markdown("""<h3><span style="color:firebrick">Diabetes Prediction</h3>""", unsafe_allow_html=True)
if not st.sidebar.checkbox("Hide", True, key='1'):
    st.markdown("""<h3><span style="color:grey">Please fill in the information in the below Diabetes Prediction Form</h3>""", unsafe_allow_html=True)
    name = st.text_input('Name:')
    age = st.number_input("Age:")
    pregnancy = st.number_input('Number of Pregnancies:')
    glucose = st.number_input("Plasma Glucose Concentration :")
    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    skin = st.number_input("Triceps skin fold thickness (mm):")
    insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    submit = st.button('Predict')
    if submit:
        prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, age]])
        if prediction == 0:
            st.write('Great News',name,'You are not diabetic')
        else:
            st.write(name," it seems like you are diabetic, please check with your doctor.")


                ####################################################################

#this is used to maximize the size of the grid
st.text(" \n")
if st.sidebar.checkbox('Maximized View'):
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
        {max_width_str}
        }}
        </style>
        """,
        unsafe_allow_html=True,
        )
