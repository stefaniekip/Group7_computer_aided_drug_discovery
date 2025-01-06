import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
from umap import UMAP


class data_preparation:
    def __init__(self, file, train_or_test, train_data = None, test_data = None, descriptors = None, n_components = None, umap_data_train = None, umap_data_test = None, descriptors_train = None, descriptors_test = None):
        self.file = file
        self.train_or_test = train_or_test
        self.train_data = train_data
        self.test_data = test_data
        self.descriptors = descriptors
        self.n_components = n_components
        self.umap_data_train = umap_data_train
        self.umap_data_test = umap_data_test
        self.descriptors_train = descriptors_train
        self.descriptors_test = descriptors_test

    def file_preparation(self):
        """
        This function reads the file and extracts the smiles.

        Returns:
            DataFrame: Cleaned data
        """
        data = pd.read_csv(self.file)

        # Clean the dataset
        # Remove duplicate molecules from dataset
        cleaned_data = data.drop_duplicates(subset='SMILES_canonical', keep='first') # Finds the duplicates in smiles and keeps the first instance

        if self.train_or_test == 'Train':
            # Make sure that the results are 0 or 1 otherwise remove molecule
            cleaned_data = cleaned_data[cleaned_data['target_feature'].isin([0, 1])]

        # Check for invalid smiles
        invalid_smiles = []
        
        for index, row in cleaned_data.iterrows(): # Iterate over the rows of the dataframe
            molecule = row['SMILES_canonical']
            mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
            if mol is None: # If the SMILES cannot be converted to an RDKit Molecule append to invalid_smiles
                invalid_smiles.append(row['SMILES_canonical']) 
  
        cleaned_data = cleaned_data.loc[~cleaned_data['SMILES_canonical'].isin(invalid_smiles)] # Take out all molecules with invalid smiles

        return cleaned_data

    def get_descriptors(self):
        '''
        Computes chemical descriptors for the molecules.

        Returns:    
            Dataframe: Each molecule with computed descriptors. 
        '''
        descriptor_data = [] # Initializes a list that will hold all descriptors

        descriptor_names = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in descriptor_names
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names) # Initializes the calculater with the wanted descriptors

        if self.train_or_test == 'Test':
            data = self.test_data
        else:
            data = self.train_data

        # Calculate the descriptors based on the SMILES
        for index, row in data.iterrows(): # Iterate over the rows of the dataframe
            molecule = row['SMILES_canonical']
            mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
            if mol is not None: # If the SMILES was succesfully converted to an RDKit Molecule
                mol_descriptors = calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
                descriptors_dict = {f"Descriptor_{i}": mol_descriptors[i] for i in range(len(mol_descriptors))} # Create a dictionary with all descriptors
                descriptors_dict['SMILES_canonical'] = row['SMILES_canonical'] # Add the SMILES  of the molecule to the dictionary
                if self.train_or_test == 'Train':
                    descriptors_dict['target_feature'] = row['target_feature'] # Add the target feature of the molecule to the dictionary
                else:
                    descriptors_dict['Unique_ID'] = row['Unique_ID'] # Add label for the molecule to the dictionary
                descriptor_data.append(descriptors_dict) # Append the dictionary for each molecule to a list to be able to create a dataframe further on
     
        # Create a new dataframe including the descriptors
        descriptors = pd.DataFrame(descriptor_data) 

        # Check whether all descriptors were calculated accurately --> what do we want to do?
        empty_descriptors = descriptors.columns[descriptors.isnull().any()] # finds columns with missing values
        descriptors = descriptors.dropna(subset=empty_descriptors) # Removes all molecules with missing values in the descriptors

        return(descriptors)
    
    def normalize_data(self):
        '''
        Normalizes data, excluding binary and identifier columns.

        Returns:
            DataFrame: Each molecule with normalized descriptors.
        '''
        # check which columns should be excluded from normalization
        NOT_normalize_columns = []
        for column in self.descriptors.columns:
            if self.train_or_test == 'Train':
                if self.descriptors[column].isin([0, 1]).all() or column == 'SMILES_canonical': # Binary columns and label columns are excluded
                    NOT_normalize_columns.append(column)
            else:
                if self.descriptors[column].isin([0, 1]).all() or column == 'Unique_ID' or column == 'SMILES_canonical': # Binary columns and label columns are excluded
                    NOT_normalize_columns.append(column)

        # Select columns that are not binary
        columns_to_normalize = []
        for column in self.descriptors.columns:
            if column not in NOT_normalize_columns: # If the columns are not part of the not normalize columns they should be normalized
                columns_to_normalize.append(column)

        self.descriptors[columns_to_normalize] = normalize(self.descriptors[columns_to_normalize], axis=0, norm='l2')
 
        return self.descriptors
    
    def umap_preparation(self):
        '''
        Applies UMAP to the data.

        Returns:
            Array: with (n_samples, n_components)
        '''
        #Initialize the UMAP reducer by the number of components
        reducer = UMAP(self.n_components)
        if self.train_or_test == 'Train':
            data = self.descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1)
        else:
            data = self.descriptors_test.drop(['Unique_ID', 'SMILES_canonical'], axis=1)

        #Scale the data                     
        scaled_train_data = StandardScaler().fit_transform(data)

        #Apply the dimension reduction
        embedding = reducer.fit_transform(scaled_train_data)

        return embedding

    def umap(self):
        '''
        Converts the UMAP data into a dataframe to make it compatible with the neural network.

        Returns:
            Tuple: a tuple containing two dataframes. The first dataframe is the training data after applying UMAP. The second dataframe is the test data after applying UMAP. 
        '''
        #Normalize the umap data
        data_train = normalize(self.umap_data_train)
        data_test = normalize(self.umap_data_test)

        # Convert UMAP data to DataFrame for compatibility with neural network
        column_name_train = []
        for index in range(data_train.shape[1]):
            column_name_train.append("UMAP_" + str(index))
        umap_train_df = pd.DataFrame(data_train, columns=column_name_train)
        umap_train_df['target_feature'] = self.descriptors_train['target_feature'].values

        column_name_test = []
        for index in range(data_test.shape[1]):
            column_name_test.append("UMAP_" + str(index))
        umap_test_df = pd.DataFrame(data_test, columns=column_name_test)

        #Make sure that the DataFrames have the same number of rows
        if self.descriptors_test.shape[0] != umap_test_df.shape[0]:
            umap_test_df = umap_test_df.iloc[:self.descriptors_test.shape[0]]
    
        umap_test_df['Unique_ID'] = self.descriptors_test['Unique_ID'].values  # Add Unique_ID for results

        return umap_train_df, umap_test_df


class neural_network:
    def __init__(self, descriptors_train, descriptors_test, model, umap, umap_data_test = None, umap_data_train = None):
        self.descriptors_train = descriptors_train
        self.descriptors_test = descriptors_test
        self.model = model
        self.umap = umap
        self.umap_data_test = umap_data_test
        self.umap_data_train = umap_data_train

    def train_model(self):
        '''
        Trains the model with either UMAP-transformed data or original descriptors.
        '''
        if self.umap:
            X = self.umap_data_train.drop(['target_feature'], axis = 1) # Get all the feature data
            y = self.umap_data_train['target_feature'] # Get all the target values 
        else:
            X = self.descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
            y = self.descriptors_train['target_feature'] # Get all the target values

        # Fit the neural network and predict outcomes
        self.model.fit(X,y)

    def predict_outcome(self):
        '''
        Predicts using the trained model and saves these results to a CSV file.
        '''
        if self.umap:
            X_test = self.umap_data_test.drop(['Unique_ID'], axis = 1) # Get all the feature data
            filename = "molecule_predictions_umap.csv" #To get the right filename
        else:
            X_test = self.descriptors_test.drop(['Unique_ID', 'SMILES_canonical'], axis=1) # Get all the feature data
            filename = "molecule_predictions.csv"
        
        test_predictions = self.model.predict(X_test)

        # Create a csv file from the predictions to test in Kaggle
        results = pd.DataFrame({'Unique_ID': self.descriptors_test['Unique_ID'], 'prediction': test_predictions})
        results.to_csv(filename, index=False)
        print("File saved at:", os.path.abspath(filename))

    def accuracy_train_data(self):
        '''
        Calculates and prints the accuracy of the model on the dataset given.
         
        It does this by splitting the dataset into training and testing datasets. Training the model on the training dataset and testing it on the test dataset. 
        This can be done on UMAP data as well if UMAP is applied.
        '''
        if self.umap:
            X = self.umap_data_train.drop(['target_feature'], axis=1) # Get all the feature data
            y = self.umap_data_train['target_feature'] # Get all the target values
        else:
            X = self.descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
            y = self.descriptors_train['target_feature'] # Get all the target values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # Split the data, random state can be removed for final hand in

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        print("Balanced accuracy:", balanced_accuracy_score(y_test, predictions), ', Umap is', self.umap)

    def cross_validation(self):
        '''
        Calculates and prints the cross validation of the model on the dataset given.
        '''
        if self.umap:
            X = self.umap_data_train.drop(['target_feature'], axis=1) # Get all the feature data
            y = self.umap_data_train['target_feature'] # Get all the target values
        else:
            X = self.descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
            y = self.descriptors_train['target_feature'] # Get all the target values
        
        scores = cross_val_score(self.model, X, y, cv=5, scoring='balanced_accuracy') #Calculating the cross-validation by using balanced accuracy  
        print("Cross validation:", scores.mean(), ", Umap is", self.umap)


def drd3_model(train_file, test_file, model, model_umap, n_components):
    '''
    Running all the classes and functions. 
    '''
    train_data_prep = data_preparation(train_file, 'Train')
    train_data = train_data_prep.file_preparation()
    test_data_prep = data_preparation(test_file, 'Test')
    test_data = test_data_prep.file_preparation()

    descriptors_train_prep = data_preparation(train_file, 'Train', train_data = train_data, test_data = test_data)
    descriptors_train = descriptors_train_prep.get_descriptors()
    descriptors_test_prep = data_preparation(test_file, 'Test', train_data = train_data, test_data=test_data)
    descriptors_test = descriptors_test_prep.get_descriptors()

    normalized_train_prep = data_preparation(train_file, 'Train', descriptors = descriptors_train)
    normalized_train_data = normalized_train_prep.normalize_data()
    normalized_test_prep = data_preparation(test_file, 'Test', descriptors=descriptors_test)
    normalized_test_data = normalized_test_prep.normalize_data()

    neural_network_prep = neural_network(normalized_train_data, normalized_test_data, model, umap = False)
    train_model = neural_network_prep.train_model()
    predict_outcome = neural_network_prep.predict_outcome()
    accuracy = neural_network_prep.accuracy_train_data()
    cross_validation = neural_network_prep.cross_validation()

    umap_prep_train = data_preparation(train_file, 'Train', n_components = n_components, descriptors_train=normalized_train_data)
    umap_train = umap_prep_train.umap_preparation()
    umap_prep_test = data_preparation(test_file, 'Test', n_components = n_components, descriptors_test = normalized_test_data)
    umap_test = umap_prep_test.umap_preparation()
    umap_prep = data_preparation(test_file, 'Test', umap_data_train = umap_train, umap_data_test = umap_test, descriptors_train = normalized_train_data, descriptors_test = normalized_test_data)
    umap_data_train, umap_data_test = umap_prep.umap()

    neural_network_umap = neural_network(normalized_train_data, normalized_test_data, model_umap, umap = True, umap_data_train = umap_data_train, umap_data_test = umap_data_test)
    neural_network_umap.train_model()
    neural_network_umap.predict_outcome()
    neural_network_umap.accuracy_train_data()
    neural_network_umap.cross_validation()