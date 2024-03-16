# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from azureml.core import Workspace, Experiment
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
import joblib

# Load dataset
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your dataset file path

# Splitting the dataset into features and target variable
X = df.drop(columns=['target_column'])  # Replace 'target_column' with the name of your target variable
y = df['target_column']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Azure ML Workspace configuration
ws = Workspace.from_config()

# Register the model in Azure ML Workspace
model = Model.register(workspace=ws,
                       model_path="model.pkl",
                       model_name="your_model_name",  # Replace 'your_model_name' with your desired model name
                       description="Model for Power BI integration")

# Define the environment
env = Environment.from_conda_specification(name="env", file_path="environment.yml")  # Provide path to your environment file

# Define inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Deploy the model as a web service on Azure Container Instance
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
service = Model.deploy(workspace=ws,
                       name="your_service_name",  # Replace 'your_service_name' with your desired service name
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)
