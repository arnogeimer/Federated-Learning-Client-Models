# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:53:18 2022

@author: arno.geimer
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:55:59 2022

@author: arno.geimer
"""
import warnings
import flwr as fl
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)

### The Preprocessing
log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate=False), StandardScaler()
)
linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        ("binned_numeric", KBinsDiscretizer(n_bins=10), ["VehAge", "DrivAge"]),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
        ),
    ],
    remainder="drop",
)

class GLM():
    def __init__(self) -> None:
        self.model= Pipeline(
            [
                ("preprocessor", linear_model_preprocessor),
                ("regressor", PoissonRegressor(alpha=1e-12,warm_start=True)),
            ]
        )

def train(glm, trainloader, epochs): 
    glm.model.named_steps['regressor'].max_iter = epochs
    features,targets,sample_weights=trainloader
    glm.model.fit(features, targets, regressor__sample_weight=sample_weights)          

def test(glm, testloader):
    features,targets,sample_weights=testloader
    exposure = features["Exposure"].values
    y_pred=glm.model.predict(features)
    mse = mean_squared_error(
        targets, y_pred, sample_weight=sample_weights
    )
    mae = mean_absolute_error(
        targets, y_pred, sample_weight=sample_weights
    )
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )
    mpd=mean_poisson_deviance(
        targets[mask],
        y_pred[mask],
        sample_weight=sample_weights[mask],
    )
    accuracy=1337#features['ClaimNb'].sum()/np.sum(y_pred * exposure)
    print(f"MPD:{round(mpd,7)}")
    return mpd,accuracy

def load_data():
    df=fetch_openml(data_id=41214, as_frame=True).frame
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    trainset,testset=train_test_split(df,train_size=.1,test_size=.5)
    return (trainset,trainset["Frequency"],trainset["Exposure"]), (testset,testset["Frequency"],testset["Exposure"])

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

glm = GLM()
trainloader, testloader = load_data()
# We have to train the model once to initialize model.coef_ and model.intercept_
train(glm, trainloader, 100)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [glm.model.named_steps['regressor'].coef_, glm.model.named_steps['regressor'].intercept_]

    def set_parameters(self, parameters):
        coefficients=parameters[0]
        intercept = parameters[1]
        glm.model.named_steps['regressor'].coef_=coefficients
        glm.model.named_steps['regressor'].intercept_=intercept

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(glm, trainloader, epochs=1)
        return self.get_parameters(config),len(trainloader[0]), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(glm, testloader)
        return loss,len(testloader[0]),{"accuracy": accuracy}


print("Client starting")

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

print("Client shutting down")

# Local predictor

glm_local = GLM()
train(glm_local, trainloader, 500)
print("Local model over testset:")
test(glm_local, testloader)