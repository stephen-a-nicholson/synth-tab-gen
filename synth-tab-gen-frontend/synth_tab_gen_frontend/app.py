"""Contains the Streamlit app for SynthTabGen."""

import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

# Configuration
API_URL = "http://localhost:8000"

# Page setup
st.set_page_config(page_title="SynthTabGen", page_icon="📊", layout="wide")

st.title("SynthTabGen")
st.subheader("GPU-Accelerated Synthetic Data for ETL Testing")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Upload Dataset",
        "Train Model",
        "Generate Data",
        "View Datasets",
        "View Models",
    ],
)


def format_time(seconds):
    """Format time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    if seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)} min {int(seconds)} sec"
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(hours)} hr {int(minutes)} min"


def poll_job_status(job_id):
    """Poll for job status until completion or failure"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        try:
            response = requests.get(f"{API_URL}/jobs/{job_id}")
            if response.status_code == 200:
                job_data = response.json()["data"]

                if job_data["status"] == "completed":
                    progress_bar.progress(100)
                    status_text.success("Task completed!")
                    return job_data["result"]
                elif job_data["status"] == "failed":
                    progress_bar.progress(100)
                    status_text.error(
                        f"Task failed: {job_data.get('error', 'Unknown error')}"
                    )
                    return None
                else:
                    # Update progress
                    progress = job_data.get("progress", 0)
                    progress_bar.progress(int(progress))
                    status_text.info(
                        f"Status: {job_data['status']} - {progress:.0f}%"
                    )
                    time.sleep(1)
            else:
                status_text.error("Failed to get job status")
                return None
        except Exception as e:
            status_text.error(f"Error checking job status: {str(e)}")
            return None


def display_data_preview(df, title="Data Preview"):
    """Display a preview of the dataframe with statistics"""
    st.subheader(title)

    # Display preview
    st.dataframe(df.head(10))

    # Display statistics
    st.subheader("Data Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Missing Values", df.isna().sum().sum())

    with col2:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Safer way to detect date columns
        date_cols = []
        for col in categorical_cols:
            # Try to convert each column individually
            try:
                if pd.to_datetime(df[col], errors="coerce").notna().any():
                    date_cols.append(col)
            except:
                pass

        st.metric("Numeric Columns", len(numeric_cols))
        st.metric("Categorical Columns", len(categorical_cols))
        st.metric("Date Columns", len(date_cols))


def display_data_distribution(df, column_name):
    """Display distribution of a single column"""
    if df[column_name].dtype in ["int64", "float64"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[column_name], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts = (
            df[column_name].value_counts().sort_values(ascending=False)
        )
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)


# Page: Upload Dataset
if page == "Upload Dataset":
    st.header("Upload Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the file and display preview
        df = pd.read_csv(uploaded_file)
        display_data_preview(df, "Original Data Preview")

        # Upload button
        if st.button("Upload Dataset"):
            # Prepare file for upload
            files = {"file": uploaded_file}

            with st.spinner("Uploading dataset..."):
                try:
                    response = requests.post(
                        f"{API_URL}/datasets/upload", files=files
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Dataset uploaded successfully!")
                        st.json(result)
                    else:
                        st.error(f"Error uploading dataset: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Page: Train Model
elif page == "Train Model":
    st.header("Train Synthetic Data Model")

    # Get available datasets
    try:
        response = requests.get(f"{API_URL}/datasets")
        if response.status_code == 200:
            datasets = response.json()["data"]["datasets"]

            if not datasets:
                st.warning(
                    "No datasets available. Please upload a dataset first."
                )
            else:
                # Dataset selection
                dataset_options = {
                    f"{d['filename']} ({d['rows']} rows)": d["dataset_id"]
                    for d in datasets
                }
                selected_dataset = st.selectbox(
                    "Select Dataset", list(dataset_options.keys())
                )
                dataset_id = dataset_options[selected_dataset]

                # Model configuration
                st.subheader("Model Configuration")

                model_type = st.selectbox(
                    "Select Model Type",
                    [
                        "CTGAN (Deep Learning)",
                        "TVAE (Deep Learning)",
                        "GaussianCopula (Statistical)",
                    ],
                )

                # Convert to API model type
                if "CTGAN" in model_type:
                    api_model_type = "CTGAN"
                elif "TVAE" in model_type:
                    api_model_type = "TVAE"
                else:
                    api_model_type = "GaussianCopula"

                # Model hyperparameters
                col1, col2 = st.columns(2)

                with col1:
                    if "Deep Learning" in model_type:
                        epochs = st.slider("Training Epochs", 10, 500, 100)
                        batch_size = st.slider("Batch Size", 100, 1000, 500)
                    else:
                        epochs = 100
                        batch_size = 500

                with col2:
                    use_gpu = st.checkbox("Use GPU Acceleration", value=True)
                    if use_gpu:
                        st.info(
                            "GPU acceleration can provide 5-15x faster training"
                        )

                # Train button
                if st.button("Train Model"):
                    # Prepare configuration
                    config = {
                        "model_type": api_model_type,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "use_gpu": use_gpu,
                    }

                    with st.spinner("Starting model training..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/models/train/{dataset_id}",
                                json=config,
                            )

                            if response.status_code == 200:
                                result = response.json()
                                job_id = result["data"]["job_id"]

                                st.info(
                                    f"Training started with job ID: {job_id}"
                                )

                                # Poll for job status
                                result = poll_job_status(job_id)

                                if result:
                                    st.success(
                                        (
                                            f"Model trained successfully!"
                                            f"Model ID: {result['model_id']}"
                                        )
                                    )
                            else:
                                st.error(
                                    f"Error starting training: {response.text}"
                                )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        else:
            st.error(f"Error fetching datasets: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Page: Generate Data
elif page == "Generate Data":
    st.header("Generate Synthetic Data")

    # Get available models
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models = response.json()["data"]["models"]

            if not models:
                st.warning("No models available. Please train a model first.")
            else:
                # Model selection
                model_options = {
                    f"{m['model_type']} Model ({m['created_at']})": m[
                        "model_id"
                    ]
                    for m in models
                }
                selected_model = st.selectbox(
                    "Select Model", list(model_options.keys())
                )
                model_id = model_options[selected_model]

                # Generation configuration
                st.subheader("Generation Configuration")

                col1, col2 = st.columns(2)

                with col1:
                    num_rows = st.number_input(
                        "Number of Rows to Generate", 100, 1000000, 10000
                    )
                    output_format = st.selectbox(
                        "Output Format", ["CSV", "JSON", "Parquet", "SQL"]
                    )

                with col2:
                    include_metrics = st.checkbox(
                        "Include Quality Metrics", value=True
                    )

                # Data quality issues
                st.subheader("Data Quality Issues")

                col1, col2, col3 = st.columns(3)

                with col1:
                    missing_values = st.slider("Missing Values (%)", 0, 30, 0)

                with col2:
                    duplicates = st.slider("Duplicate Records (%)", 0, 20, 0)

                with col3:
                    outliers = st.slider("Outliers (%)", 0, 10, 0)

                # Generate button
                if st.button("Generate Synthetic Data"):
                    # Prepare configuration
                    config = {
                        "num_rows": num_rows,
                        "include_quality_metrics": include_metrics,
                        "output_format": output_format.lower(),
                        "missing_values_percent": missing_values,
                        "duplicates_percent": duplicates,
                        "outliers_percent": outliers,
                    }

                    with st.spinner("Starting data generation..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/jobs/generate-data/{model_id}",
                                json=config,
                            )

                            if response.status_code == 200:
                                result = response.json()
                                job_id = result["data"]["job_id"]

                                st.info(
                                    f"Generation started with job ID: {job_id}"
                                )

                                # Poll for job status
                                result = poll_job_status(job_id)

                                if result:
                                    dataset_id = result["dataset_id"]
                                    st.success(
                                        f"Data generated successfully! Dataset ID: {dataset_id}"
                                    )

                                    # Display preview of generated data
                                    response = requests.get(
                                        f"{API_URL}/jobs/data/{dataset_id}"
                                    )
                                    if response.status_code == 200:
                                        preview_data = response.json()["data"][
                                            "preview"
                                        ]
                                        df_preview = pd.DataFrame(preview_data)

                                        st.subheader("Generated Data Preview")
                                        st.dataframe(df_preview)

                                        # Download button
                                        output_format_lower = (
                                            output_format.lower()
                                        )
                                        if output_format_lower == "csv":
                                            csv = df_preview.to_csv(
                                                index=False
                                            )
                                            st.download_button(
                                                "Download Sample CSV",
                                                csv,
                                                f"synthetic_data_sample.csv",
                                                "text/csv",
                                                key="download-csv",
                                            )
                            else:
                                st.error(
                                    f"Error starting generation: {response.text}"
                                )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        else:
            st.error(f"Error fetching models: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Page: View Datasets
elif page == "View Datasets":
    st.header("Available Datasets")

    if st.button("Refresh Datasets"):
        st.experimental_rerun()

    # Get available datasets
    try:
        response = requests.get(f"{API_URL}/datasets")
        if response.status_code == 200:
            datasets = response.json()["data"]["datasets"]

            if not datasets:
                st.warning(
                    "No datasets available. Please upload a dataset first."
                )
            else:
                # Display datasets in a table
                df_datasets = pd.DataFrame(datasets)
                st.dataframe(df_datasets)

                # Dataset selection for more details
                dataset_options = {
                    f"{d['filename']} ({d['rows']} rows)": d["dataset_id"]
                    for d in datasets
                }
                selected_dataset = st.selectbox(
                    "Select Dataset for Details", list(dataset_options.keys())
                )
                dataset_id = dataset_options[selected_dataset]

                # Get dataset details
                response = requests.get(f"{API_URL}/datasets/{dataset_id}")
                if response.status_code == 200:
                    dataset = response.json()["data"]

                    st.subheader(f"Dataset: {dataset['filename']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", dataset["rows"])
                    col2.metric("Columns", len(dataset["columns"]))
                    col3.metric("Created", dataset["created_at"].split("T")[0])

                    st.subheader("Columns")
                    st.write(dataset["columns"])
        else:
            st.error(f"Error fetching datasets: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Page: View Models
elif page == "View Models":
    st.header("Available Models")

    if st.button("Refresh Models"):
        st.experimental_rerun()

    # Get available models
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            models = response.json()["data"]["models"]

            if not models:
                st.warning("No models available. Please train a model first.")
            else:
                # Display models in a table
                df_models = pd.DataFrame(models)
                st.dataframe(df_models)

                # Model selection for more details
                model_options = {
                    f"{m['model_type']} Model ({m['created_at']})": m[
                        "model_id"
                    ]
                    for m in models
                }
                selected_model = st.selectbox(
                    "Select Model for Details", list(model_options.keys())
                )
                model_id = model_options[selected_model]

                # Get model details
                response = requests.get(f"{API_URL}/models/{model_id}")
                if response.status_code == 200:
                    model = response.json()["data"]

                    st.subheader(f"Model: {model['model_type']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Type", model["model_type"])
                    col2.metric("Created", model["created_at"].split("T")[0])

                    if "config" in model:
                        st.subheader("Configuration")
                        st.json(model["config"])
        else:
            st.error(f"Error fetching models: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Footer
st.markdown("---")
st.markdown("SynthTabGen - Synthetic Data Generation for ETL Testing")
