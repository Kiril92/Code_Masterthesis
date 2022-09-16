# Setup
import numpy as np
from numpy import shape
import pandas as pd
import matplotlib.pyplot as plt

import gudhi as gd 
import gudhi.representations

from math import *
import random
import dill
import glob
import gc

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import concatenate

def create_subsample(dataframe, size, replace = False, verbose = False):
    """
    Given a 2D numpy array, this function creates a subset of the rows and columns of this array
    """
    # Extract number of rows in the array
    a = len(dataframe)
    
    if verbose:
        print("Sampling", size, "from", a)
    
    # Sampling equally distributed integers
    indizes = np.sort(np.random.choice(a, size, replace=replace))
    
    # Select rows and columns given the sampled indizes
    dataframe_short = dataframe[:, indizes]
    dataframe_short = dataframe_short[indizes, :]
    
    return dataframe_short

def calc_t_sigma(sample_index, M_sqr, X_sqr, M, verbose = False):
    """
    Calculates the t_sigma matrix
    """
    
    # Cast to a numpy array for compatibility purpose
    M_sqr = np.asarray(M_sqr)
    X_sqr = np.asarray(X_sqr)
    
    # Extract number of columns
    ncols = shape(X_sqr)[1]
    
    # Preallocation of the matrix
    t_sigma = np.zeros((ncols, ncols))
    
    # Calculation of lower triangular matrix
    for i in range(ncols):
        if verbose:
            print(i, " von ", ncols)
        
        t_sigma[i,0:i] = np.sqrt(M_sqr[i,0:i] + np.square((X_sqr[sample_index,i] - X_sqr[sample_index,0:i])/M[i,0:i]) + 2*X_sqr[sample_index,i] + 2*X_sqr[sample_index,0:i])/2
  
    # Mirror on the diagonal
    t_sigma = t_sigma + np.transpose(t_sigma)
    
    # When the distance is 0, the t_sigma entries becomes NaN. These are changed to zeroes
    t_sigma[np.isnan(t_sigma)] = 0
    
    return t_sigma

def calc_pl(simplex_tree, dimension=1, resolution = 100, num_landscapes = 5, sample_range=[nan, nan]):
    """
    Given a simple tree, this function calculates the according persistence landscapes.
    """
    
    # Initialize PL-representation
    LS = gd.representations.Landscape(resolution = resolution,
                                      num_landscapes = num_landscapes,
                                      sample_range = sample_range)
    
    # For dimension 0 we have to remove the inf entries
    if dimension==1:
        pers_intervals = simplex_tree.persistence_intervals_in_dimension(1)
    
    elif dimension==0:
        pers_intervals = simplex_tree.persistence_intervals_in_dimension(0)
        
        # remove rows with Inf-entries, to avoid errors when plotting
        pers_intervals = pers_intervals[pers_intervals[:,1] != Inf, :]
    
    else:
        raise ValueError('Parameter "dimension" should be 0 or 1')
        
    L = LS.fit_transform([pers_intervals])
    return L[0]


def calc_persistence_oneperson(t_sigma, n_subsample = None, n_persistences = 1, concat_buckets=False, verbose=False):
    """
    Given the t_sigma matrix, this function calculates multiple subsamples and the according vietoris-rips complexes as well as the persistences.
    """
    
    # Initialize list for persistences
    persistence_oneperson = []
    
    for index_persistence in range(n_persistences):
        if(verbose & (index_persistence%20 == 0)):
            print("Calculating Persistenz ", index_persistence+1, " of ", n_persistences)
        
        # Subsampling
        if n_subsample is not None:
            t_sigma_short = create_subsample(t_sigma,
                                             size = n_subsample)
        else:
            t_sigma_short = t_sigma
        
        # RipsComplex
        rips_complex = gd.RipsComplex(
            distance_matrix = t_sigma_short
        ) 
        
        # Persistence
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
        BarCodes_Rips = simplex_tree.persistence()
        
        persistence_oneperson.insert(index_persistence, BarCodes_Rips)
        
    return persistence_oneperson

def extract_homology(persistence, homology = 1, remove_inf = True):
    """
    Given an array with persistences from multiple homology groups, this function extract the entries according to one specific homology group.
    """
    # Cast to Numpy-Array to allow indexing
    resulting_persistence = np.asarray(a = persistence, dtype = object)
    
    if len(persistence) == 1:
        resulting_persistence = resulting_persistence[0]
        
    resulting_persistence = resulting_persistence[resulting_persistence[:,0] == homology]
    
    # If there is no element e.g. in homology 1, return an empty list
    if len(resulting_persistence) == 0:
        return []
    
    # Remove inf-entries in persistences
    resulting_persistence = resulting_persistence[np.isfinite(np.stack(resulting_persistence[:,1])[:,1])]
    
    return resulting_persistence

def pl_from_persistence(persistence, resolution = 1000, num_landscapes = 10, homology = 1, sample_range=[nan, nan]):
    """
    Given one persistence, this function calculates the according persistence landscape
    """
    
    # Filter one specific homology group
    persistence = extract_homology(persistence, homology = homology, remove_inf = True)
    
    if len(persistence) == 0:
        return np.full((num_landscapes*resolution), nan)
    
    # formatting
    persistence = np.stack(persistence[:,1], axis=0)
    
    # Sorting in ascending order by moment of birth
    persistence = persistence[persistence[:, 0].argsort()]
    
    LS = gd.representations.Landscape(resolution = resolution,
                                      num_landscapes = num_landscapes,
                                      sample_range = sample_range).fit_transform([persistence])[0]
    
    return LS

def pi_from_persistence(persistence, bandwidth = 1.0, resolution = [20, 20], weight= lambda x: 1, homology = 1, im_range = [nan, nan, nan, nan]):
    """
    Given one persistence, this function calculates the according persistence image
    """
    
    # Filter one specific homology group
    persistence = extract_homology(persistence, homology = homology, remove_inf = True)
    
    if len(persistence) == 0:
        return np.full((resolution[0]*resolution[1]), nan)

    # Formatting
    persistence = np.stack(persistence[:,1], axis=0)
    
    # Sorting in ascending order by moment of birth
    persistence = persistence[persistence[:, 0].argsort()]
    
    PI = gd.representations.PersistenceImage(bandwidth = bandwidth,
                                             resolution = resolution,
                                             im_range = im_range,
                                             weight = weight).fit_transform([persistence])[0]
    
    return PI

def plot_pl(pl, resolution, axes = None,  num_landscapes=5, title='Landscape'):
    """
    Plotting the persistence landscapes
    """
    if axes is not None:
        for i in range(num_landscapes):
            axes.plot(pl[i*resolution : (i+1)*resolution])
    
    else:
        for i in range(num_landscapes):
            plt.plot(pl[i*resolution : (i+1)*resolution])
        plt.title(title)
        plt.show()
        
def plot_pi(pi, resolution = [20, 20], cmap = None, axes = None, title = "Persistence Image"):
    """
    Plotting the persistence images
    """
    if axes is not None:
        axes.imshow(np.flip(np.reshape(pi, resolution), 0), cmap = cmap)
    
    else:
        plt.imshow(np.flip(np.reshape(pi, resolution), 0), cmap = cmap)
        plt.title(title)
        plt.show()

def calc_avg_PL_onesubject(all_persistences_onesubject,
                           resolution = 1000,
                           num_landscapes = 10,
                           homology = 1,
                           sample_range=[nan, nan]):
    """
    Given an array of multiple persistences, this function first calculates the single persistence landscapes to calculate the mean persistence landscape
    """
    avg_pl_all_persistences = []
    
    # Calculate persistence landscape for every single persistence
    for index_persistence in range(len(all_persistences_onesubject)):
        avg_pl_all_persistences.insert(index_persistence, pl_from_persistence(all_persistences_onesubject[index_persistence],
                                                                              resolution = resolution,
                                                                              num_landscapes = num_landscapes,
                                                                              homology = homology,
                                                                              sample_range = sample_range
                                                                             ))
    
    # Averaging over all persistences
    avg_pl = np.nanmean(avg_pl_all_persistences, axis=0)

    return avg_pl


def get_minima(persistence, homology = 1):
    """
    Given one persistence, this function finds the lowest birth-value
    """
    tmp = extract_homology(persistence, homology = homology)

    # If there is no element in the homology group, return an empty list
    if len(tmp) == 0:
        return nan
        
    # Formatting array of tuples into a 2D numpy array
    tmp_array = np.stack(tmp[:,1])

    # Find minima of birth values in the persistence
    min_value_in_persistence = np.amin(tmp_array[:,0])

    return min_value_in_persistence

def get_all_minima(all_persistences, homology = 1):
    """
    Given an array of multiple persistences, this function finds the lowest birth-value for each peristence
    """
    all_minima = []
    
    for index_persistence in range(len(all_persistences)):
        one_minima = get_minima(all_persistences[index_persistence])
        
        all_minima.append(one_minima)
        
    return np.asarray(all_minima)

def get_all_maxima(persistences_onesubject, homology = 1):
    """
    Given multiple persistences of one subject, this function extracts the maximum of all death values
    """
    all_maxima = []
    
    for index_persistence in range(len(persistences_onesubject)):
        tmp = extract_homology(persistences_onesubject[index_persistence], homology = homology)

        # Formatting array of tuples into a 2D numpy array
        tmp_array = np.stack(tmp[:,1])
        
        # Find maxima of death values in persistence
        max_value_in_persistence = np.amax(tmp_array[:,1])
        
        # Add maxima of one persistence to the list
        all_maxima.insert(index_persistence, max_value_in_persistence)
        
    return all_maxima

def get_max_persistence_value(pers_intervals, depth = 2):
    
    # Rowbind der Listenelemente
    for i in range(depth):
        pers_intervals = np.concatenate(pers_intervals, axis=0)
    
    max_value_index = np.where(pers_intervals[:,0] == np.nanmax(pers_intervals[:,0]))[0][0]
    print("Max indizes are: ", max_value_index)
    print("Max values are: ", pers_intervals[max_value_index,:])
    
    max_sample_size = pers_intervals[max_value_index,0]
    
    return max_sample_size

def save_file(file, x):
    """
    Saves a variable into a file
    """
    print("Saving " + file)
    with open(file, 'wb') as f:
        dill.dump(x, f)

def load_file(file):
    """
    Loads a file into a variable
    """
    with open(file, 'rb') as f:
        loaded_data = dill.load(f)
    return loaded_data

def calc_avg_PL_from_all_persistences(all_persistences, homology = 1, resolution = 1000, num_landscapes = 10, scaling = "within_subjects", verbose = True):
    """
    Given multiple persistences for one subject, this function calculates the relevant parameter for the scaling of persistence landscapes as well as the persistence landscapes itself. Finally the mean of all persistance landscapes is returned.
    """
    
    # check for between_subject scaling
    scaling_calculated = False
    
    avg_pl_allsubjects = []
    
    for index_subject in range(len(all_persistences)):
        if verbose:
            print("Calculating PL for sample no.", index_subject)
            
        avg_pl_onesubject = []
        
        if not scaling_calculated:
            if (scaling == "between_subjects"):
                
                print("Extract maximum from all persistences of all subjects")
                maxima_allsubjects = []
                for index_maxima_subject in range(len(all_persistences)):
                    # Extrahiere erst alle Maxime je Persistenz und dann globales Maxima
                    all_maxima_onesubject = get_all_maxima(persistences_onesubject = all_persistences[index_maxima_subject])
                    total_maxima_onesubject = np.amax(all_maxima_onesubject)
                    maxima_allsubjects.insert(index_maxima_subject, total_maxima_onesubject)
                    
                total_max_allsubjects = np.amax(maxima_allsubjects)
                
                if verbose:
                    print("Total Maximum between all subjects ", total_max_allsubjects)
                    
                scaling_calculated = True
                
                sample_range = [0, total_max_allsubjects]
                    
            elif scaling == "within_subjects":
                # Extrahiere erst alle Maxima je Persistenz und dann globales Maxima
                maximas_per_persistence = get_all_maxima(persistences_onesubject = all_persistences[index_subject])
                total_max_subject = np.amax(maximas_per_persistence)
                
                sample_range = [0, total_max_subject]
                
            elif scaling == "unscaled":
                sample_range = [nan,nan]
                
            else:
                raise ValueError('Parameter "scaling" should be "between_subjects", "within_subjects" or "unscaled"')
        
        avg_pl_allsubjects.insert(index_subject, calc_avg_PL_onesubject(all_persistences[index_subject],
                                                                        resolution = resolution,
                                                                        num_landscapes = num_landscapes,
                                                                        homology = homology,
                                                                        sample_range = sample_range))
        
    return avg_pl_allsubjects

def get_all_maxima_all_subjects(persistences, verbose = True):
    """
    Given an array of arrays containing multiple persistences for multiple subjects, this function returns the maxima of all persistences.
    """
    # allocate results-dict
    all_maxima_all_subjects = []
    
    for index_subject in range(len(persistences)):
        
        # print progress
        if verbose:
            print(index_subject)
        
        # Get maxima for each persistence per subject
        all_maxima_onesubject_tmp = get_all_maxima(persistences_onesubject = persistences[index_subject])
        
        # Extract maxima over all persistences per subject
        all_maxima_all_subjects.insert(index_subject, np.amax(all_maxima_onesubject_tmp))
        
    return all_maxima_all_subjects
    

# Functions used in the machine learning implementation ---------------
from sklearn import metrics

def calc_accuracy(CM):
    """
    Caculate the accuracy from the confusion matrix
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return (TP+TN)/(TP+TN+FP+FN)

def calc_precision(CM):
    """
    Caculate the precision from the confusion matrix
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP/(TP+FP)

def calc_recall(CM):
    """
    Caculate the recall from the confusion matrix
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP/(TP+FN)

def calc_TPR(CM):
    """
    Caculate the true positive rate from the confusion matrix
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    # Sensitivity, hit rate, recall, or true positive rate
    return TP/(TP+FN)
    
def calc_TNR(CM):
    """
    Caculate the true negative rate from the confusion matrix
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    return TN/(TN+FP)
    
def evaluation(y_test, y_pred, filename_csv = None):
    CM = metrics.confusion_matrix(y_test, y_pred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    
    accuracy = calc_accuracy(CM)
    
    print("Confusion-Matrix:\n",
          CM, "\n",
         "\nTrue Negative: ", TN,
         "\nTrue Positive", TP,
         "\nFalse Negative", FN,
         "\nFalse Positive", FP)
    
    print("\nTPR:", TPR,
         "\nTNR: ", TNR)
    print("\nAccuracy:", accuracy)
    print("F1 (micro):",metrics.f1_score(y_test, y_pred, average='micro'))
    print("F1 (macro):",metrics.f1_score(y_test, y_pred, average='macro'))
    
    if filename_csv is not None:
          
        # Combine values into dictionary 
        df = pd.DataFrame([[accuracy, TPR, TNR]], columns=['accuracy', 'TPR', 'TNR'])
        
        df.to_csv(filename_csv, encoding='utf-8', index=False)
        
def combine_models(models, units_final_layer = 32):
    """
    Given 2 different neural networks, this function combines both models after their output layers to one new overall model.
    """
    tmp_model1 = models[0]()
    tmp_model2 = models[1]()
    
    # Concatenate output-layers of both models
    combinedInput = concatenate([tmp_model1.output, tmp_model2.output])
    
    # Create final- and output layer
    x = Dense(32, activation="relu")(combinedInput)
    x = Dense(1, activation="softmax")(x)
    
    # final model
    model = Model(inputs = [tmp_model1.input, tmp_model2.input],
                  outputs = x)
    
    return model
    
def test_model(model, x_train, y_train, x_test, y_test, validation_split_, learning_rate_, epochs_, verbose_ = 0):
    """
    Given a ML model and training- & test-data, this function trains the model and evaluates it on the test data.
    """
    # Initialize model and model_id
    if shape(model) == ():
        tmp_model = model()
        model_id = model.__name__
    elif len(model) == 2:
        tmp_model = combine_models(models = model)
        model_id = model[0].__name__ + " & " + model[1].__name__
    else:
        raise ValueError("Bisher nur für maximal 2 Modelle implementiert")
    
    if verbose_:
        print("Train model: ", model_id)
    
    tmp_model.compile(optimizer=optimizers.Adam(learning_rate = learning_rate_),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])
    
    # Fit model to data
    history = tmp_model.fit(x_train,
                            y_train,
                            validation_split = validation_split_,
                            epochs = epochs_,
                            verbose = verbose_)
    
    y_pred = tmp_model.predict(x_test)

    test_CM = metrics.confusion_matrix(y_test, np.rint(y_pred)) # np.rint() rundet auf ganze Zahlen
    test_accuracy = calc_accuracy(test_CM)
    test_TPR = calc_TPR(test_CM)
    test_TNR = calc_TNR(test_CM)
    
    result_scores = pd.DataFrame({'model': model_id, 
                                  #'history': history, # training history removed due to high volume
                                  'learning_rate' : learning_rate_,
                                  'val_split' : validation_split_,
                                  'epochs' : epochs_,
                                  'accuracy': test_accuracy,
                                  'TPR' : test_TPR,
                                  'TNR' : test_TNR},
                                index=[0])
    
    # Free RAM
    del tmp_model
    gc.collect()
    
    return result_scores

def create_modelgrid(models = [], models_to_combine = None, learning_rates = [0.001], validation_splits = [0.0], epochs = [1]):
    """
    Given multiple values for hyperparameters, this function creates a grid containing all possible combinations of values.
    """
    if len(models)==0:
        raise ValueError("No 'models'-input'")
        
    grid = np.array(np.meshgrid(models, models_to_combine, learning_rates, validation_splits, epochs)).T.reshape(-1,5)
    grid = pd.DataFrame(grid, columns = ['model', 'models_to_combine', 'learning_rate', 'validation_split', 'epochs'])
        
    return grid

def test_multiple_models(x_train, y_train, x_test, y_test, modelgrid, verbose=True):
    """
    This function takes training- and testdata as well as a modelgrid as input and trains all models within the modelgrid with the according values for hyperparameters.
    """
    # allocate results-dataframe
    results = pd.DataFrame(columns=['model', 'learning_rate', 'val_split', 'epochs', 'accuracy', "TPR", "TNR"])

    for rowindex_grid in range(len(modelgrid)):
        
        # print training progress
        if verbose:
            print("Training model", rowindex_grid+1, "of", len(modelgrid))
            
        # Detect whether there are models to combine
        if modelgrid.loc[rowindex_grid, "models_to_combine"] is None:
            current_model = modelgrid.loc[rowindex_grid, 'model']
        else:
            current_model = [modelgrid.loc[rowindex_grid, 'model'],
                             modelgrid.loc[rowindex_grid, 'models_to_combine']]
            
        tmp = test_model(model = current_model,
                         x_train = x_train,
                         y_train = y_train, 
                         x_test = x_test, 
                         y_test = y_test, 
                         validation_split_ = modelgrid.loc[rowindex_grid, 'validation_split'],
                         epochs_ = modelgrid.loc[rowindex_grid, 'epochs'],
                         learning_rate_ = modelgrid.loc[rowindex_grid, 'learning_rate']
                        )
        
        results = results.append(tmp, ignore_index=True)
        
    return results

# Legacy code for buckets -------------------------------------

def seperate_into_buckets(persistences, bucket_size, multiple_subjects=True, verbose=True):
    """
    Given an array of multiple persistences, this function combines persistences into buckets.
    """
    # Printing number of subjects, number of given persistences and the number of resulting buckets
    print("Detect dimensions of the entered data")
    input_n_subjects = len(persistences)
    input_n_persistences_per_subject = len(persistences[0])
    
    n_buckets = floor(input_n_persistences_per_subject/bucket_size)
    
    print("Subjects available:", input_n_subjects,
         "\nAvailable persistences per subject:", input_n_persistences_per_subject)

    # Checking whether the number of draws can be divided on the bucket size without rest
    if(input_n_persistences_per_subject%bucket_size == 0):
        print("!The", input_n_persistences_per_subject, "persistences can be seperated into", n_buckets, "buckets of size", bucket_size," without rest.")
    else:
        print("!The", input_n_persistences_per_subject, "persistences can not be seperated into", n_buckets, "buckets of size", bucket_size,"without rest.",
             "\nThe remaining", input_n_persistences_per_subject%bucket_size, "persistences are not considerd!")
    
    ### Seperating into buckets -------------------------------------------

    # Initialize list for final output
    persistences_all_subjects_in_buckets = []
    
    for index_subject in range(input_n_subjects):
        if verbose:
            print("Preparing persistence for subject ", index_subject+1, "of", input_n_subjects)
        
        # Allocate list for buckets of one subject
        all_buckets_onesubject = []
        
        for index_bucket in range(n_buckets):
            one_bucket = persistences[index_subject][(index_bucket*bucket_size) : ((index_bucket+1)*bucket_size)]
            all_buckets_onesubject.insert(index_bucket, np.concatenate(one_bucket, axis=0))
        
        persistences_all_subjects_in_buckets.insert(index_subject, all_buckets_onesubject)
    
    return persistences_all_subjects_in_buckets