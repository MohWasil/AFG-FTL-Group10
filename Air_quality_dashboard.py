import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib




# Model loading
model = joblib.load("./Random Forest_model.pkl")

# wide layout
st.set_page_config(layout="wide")

# Title of Dashboard
st.markdown(
    """
    <h1 style="text-align: center; color: white;">
        Air Quality Prediction and Classification
    </h1>
    """,
    unsafe_allow_html=True
)


# upload file in a side bar
with st.sidebar:
    st.header("Upload a file") 
    upload_file = st.sidebar.file_uploader("Upload a CSV file!", type=["csv"])


# Preprocess
def preprocess(dataset):
    dataset = dataset.drop("Unnamed: 0", axis=1)
    return dataset


# Make sure to have an uploaded file
if upload_file is not None:
    # read the data
    data = pd.read_csv(upload_file)
    
    # Display first 5 rows of the data
    with st.expander("Data preview"):        
        st.dataframe(data.head(5))
    
    # Preprocess the dataset before prediction
    data = preprocess(data)
    # predict the data
    prediction = model.predict(data)
    
    # store the prediction in a new column
    data["Predicted Air Quality"] = prediction
    
    # Display the prediction to the dashboard
    with st.expander("Predicted Air Quality"):
        st.dataframe(data)
    
    # Custome barchart color
    custom_colors = {
    "Normal": "#1f77b4",    # Blue
    "Moderate": "#ff7f0e",  # Orange
    "Abnormal": "#2ca02c",  # Green
    "Dangerous": "#d62728", # Red
    }

    # Visualization 1: Bar Chart
    bar_chart = px.bar(
        data_frame=data,
        x="Predicted Air Quality",
        color="Predicted Air Quality",
        title="Air Quality Prediction Counts",
        labels={"Predicted Air Quality": "Air Quality Type"},
        template="plotly_dark",
        color_discrete_map=custom_colors  # Vibrant color scheme
    )
    # Remove any unwanted visual effects
    bar_chart.update_traces(
        marker=dict(opacity=1, line=dict(width=0))  # No transparency, no outline
    )


    # Visualization 2: Histogram
    if "CO(GT)" in data.columns:  # Ensure CO column exists
        histogram = px.histogram(
            data_frame=data,
            x="CO(GT)",
            nbins=20,
            title="Distribution of CO Levels",
            labels={"CO(GT)": "CO Levels"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel1,  # Nice pastel colors
        )
    else:
        histogram = None

    # Visualization 3: Animated Scatter Plot
    if "CO(GT)" in data.columns and "NOx(GT)" in data.columns:
        # Ensure positive values
        data["CO(GT)"] = data["CO(GT)"].apply(lambda x: x if x > 0 else 0.1)
        data["NOx(GT)"] = data["NOx(GT)"].apply(lambda x: x if x > 0 else 0.1)       
        data["NO2(GT)"] = data["NO2(GT)"].apply(lambda x: x if x > 0 else 0.1)
        scatter = px.scatter(
            data_frame=data,
            x="CO(GT)",
            y="NOx(GT)",
            animation_frame="CO(GT)",  # Replace with a time or numeric column
            color="Predicted Air Quality",
            size="NO2(GT)",  # Bubble size
            hover_data=["Predicted Air Quality", "CO(GT)", "NO2(GT)", "NOx(GT)"],
            title="Animated Scatter Plot with CO vs NO",
            labels={"CO(GT)": "CO Levels", "NOx(GT)": "NO Levels"},
            template="plotly_dark",
        )
    else:
        scatter = None

    # Visualization 4: Circle Graph (Bubble Chart)
    if "NMHC(GT)" in data.columns and "C6H6(GT)" in data.columns and "NO2(GT)" in data.columns:
        bubble_chart = px.scatter(
            data_frame=data,
            x="NMHC(GT)",
            y="C6H6(GT)",
            size="NO2(GT)",
            color="Predicted Air Quality",
            hover_data=["Predicted Air Quality", "NMHC(GT)", "C6H6(GT)", "NO2(GT)"],
            title="Bubble Chart of NMHC vs Benzene",
            labels={"NMHC(GT)": "NMHC Levels", "C6H6(GT)": "Benzene Levels", "NO2(GT)": "NO2 Levels"},
            template="plotly_dark",
        )
    else:
        bubble_chart = None



    # Display graphs in a grid layout (two per row)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(bar_chart, use_container_width=True)
        if scatter:
            st.plotly_chart(scatter, use_container_width=True)
    with col2:
        if histogram:
            st.plotly_chart(histogram, use_container_width=True)
        if bubble_chart:
            st.plotly_chart(bubble_chart, use_container_width=True)
    
    # Total value count of predicted categories statistics
    with st.expander("Air Quality Classification Distribution"):
    # st.subheader('Air Quality Classification Distribution')
        st.write(data['Predicted Air Quality'].value_counts())

else:
    st.info("Upload a file through config", icon="✔")
    st.stop()



