from DRD3_model import drd3_model
from sklearn.neural_network import MLPClassifier
from umap import UMAP

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os

train_file = 'train.csv'
test_file = 'test.csv'

model = MLPClassifier(hidden_layer_sizes=(210, 105, 52, 26, 13, 6, 3, 1), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=100, 
                            learning_rate='constant', 
                            max_iter=500, 
                            random_state=1)

model2 = MLPClassifier(hidden_layer_sizes=(210, 105, 52, 26, 13, 6, 3, 1),    
                        activation='tanh', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=64, 
                            learning_rate='adaptive', 
                            max_iter=1000, 
                            random_state=1)


model_umap = MLPClassifier(hidden_layer_sizes=(50, 100, 25), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=100, 
                            learning_rate='constant', 
                            max_iter=500, 
                            random_state=1)

model_umap2 = MLPClassifier(hidden_layer_sizes=(200, 100, 50, 25), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.001,
                            batch_size=64, 
                            learning_rate='adaptive', 
                            max_iter=1000, 
                            random_state=1)


drd3_model(train_file, test_file, model, model_umap2, n_components = 10)



