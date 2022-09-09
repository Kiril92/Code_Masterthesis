import numpy as np
from numpy import shape
import pandas as pd

import gudhi as gd 
import gudhi.representations

from math import *
import random
import dill
import glob
import gc

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import concatenate

# Hilfsfunktion um ein Subsampling zu erstellen
def create_subsample(dataframe, size, replace = False, verbose = False):
    # Extrahiere Anzahl an Zeilen
    a = len(dataframe)
    
    if verbose:
        print("Sampling", size, "from", a)
    indizes = np.sort(np.random.choice(a, size, replace=replace))
    
    dataframe_short = dataframe[:, indizes]
    dataframe_short = dataframe_short[indizes, :]
    
    return dataframe_short

def calc_t_sigma(sample_index, M_sqr, X_sqr, M, verbose = False):
    
    M_sqr = np.asarray(M_sqr)
    X_sqr = np.asarray(X_sqr)
    
    ncols = shape(X_sqr)[1]
    
    # Allokation der Matrix
    t_sigma = np.zeros((ncols, ncols))
    
    # Berechnung der unteren Dreiecksmatrix
    for i in range(ncols):
        if verbose:
            print(i, " von ", ncols)
        
        t_sigma[i,0:i] = np.sqrt(M_sqr[i,0:i] + np.square((X_sqr[sample_index,i] - X_sqr[sample_index,0:i])/M[i,0:i]) + 2*X_sqr[sample_index,i] + 2*X_sqr[sample_index,0:i])/2
  
    # Spiegelung um die Diagonale
    t_sigma = t_sigma + np.transpose(t_sigma)
    
    # Bei einer Distanzkorreltation von 1 kommt es zu nan-Einträgen in t_sigma
    t_sigma[np.isnan(t_sigma)]=0
    
    return t_sigma

def calc_pl(simplex_tree, dimension=1, resolution = 100, num_landscapes=5, sample_range=[nan, nan]):
    
    # Initialisiere PL-Repräsentation
    LS = gd.representations.Landscape(resolution = resolution,
                                      num_landscapes = num_landscapes,
                                      sample_range = sample_range)
    
    if dimension==1:
        pers_intervals = simplex_tree.persistence_intervals_in_dimension(1)
    
    elif dimension==0:
        pers_intervals = simplex_tree.persistence_intervals_in_dimension(0)
        
        # Entferne Zeilen mit Inf-Einträgen, um Fehler beim Plotten zu vermeiden
        pers_intervals = pers_intervals[pers_intervals[:,1] != Inf, :]
    
    else:
        raise ValueError('Parameter "dimension" should be 0 or 1')
        
    L = LS.fit_transform([pers_intervals])
    return L[0]


def calc_persistence_oneperson(t_sigma, n_subsample = None, n_persistences = 1, concat_buckets=False, verbose=False):
    
    n_genes = len(t_sigma)
    
    # Initialisiere Listen für die Buckets
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

def seperate_into_buckets(persistences, bucket_size, multiple_subjects=True, verbose=True):
    
    # Ausgabe über Anzahl Subjects, Anzahl vorhandener Persistenzen und Anzahl Buckets
    print("Ermittle Dimensionen der eingegebenen Daten")
    input_n_subjects = len(persistences)
    input_n_persistences_per_subject = len(persistences[0])
    
    n_buckets = floor(input_n_persistences_per_subject/bucket_size)
    
    print("Vorhandene Probanden:", input_n_subjects,
         "\nVorhandene Persistenzen je Proband:", input_n_persistences_per_subject)

    # Prüfung, ob die Anzahl an Ziehungen auf die Bucketgröße ohne Rest aufgeteilt werden kann
    if(input_n_persistences_per_subject%bucket_size == 0):
        print("!Die", input_n_persistences_per_subject, "Persistenzen können ohne Rest auf", n_buckets, "Buckets der Größe", bucket_size,"aufgeteilt werden")
    else:
        print("!Die", input_n_persistences_per_subject, "Persistenzen können nicht ohne Rest auf", n_buckets, "Buckets der Größe", bucket_size,"aufgeteilt werden.",
             "\nDie letzten verbleibenden", input_n_persistences_per_subject%bucket_size, "Persistenzen werden nicht berücksichtigt!")
    
    ### Verteilung auf die Buckets -------------------------------------------

    # Initiiere Liste für den finalen Output
    persistences_all_subjects_in_buckets = []
    
    for index_subject in range(input_n_subjects):
        if verbose:
            print("Verarbeite Persistenzen für Proband", index_subject+1, "von", input_n_subjects)
        
        # Allokiere Liste für alle Buckets eines Probanden
        all_buckets_onesubject = []
        
        for index_bucket in range(n_buckets):
    #         print(range((index_bucket*bucket_size), (index_bucket+1)*bucket_size))
    #         one_bucket = []
            
            one_bucket = persistences[index_subject][(index_bucket*bucket_size) : ((index_bucket+1)*bucket_size)]
            all_buckets_onesubject.insert(index_bucket, np.concatenate(one_bucket, axis=0))
        
        persistences_all_subjects_in_buckets.insert(index_subject, all_buckets_onesubject)
    
    return persistences_all_subjects_in_buckets

def extract_homology(persistence, homology = 1, remove_inf = True):
    # Cast rum Numpy-Array um Indexierung zu ermöglichen
    resulting_persistence = np.asarray(a = persistence, dtype = object)
    
    if len(persistence) == 1:
        resulting_persistence = resulting_persistence[0]
        
    resulting_persistence = resulting_persistence[resulting_persistence[:,0] == homology]
    
    # If there is no element e.g. in homology 1, return an empty list
    if len(resulting_persistence) == 0:
        return []
    
    # Filtern von inf-Einträgen in Persistenz
    resulting_persistence = resulting_persistence[np.isfinite(np.stack(resulting_persistence[:,1])[:,1])]
    
    return resulting_persistence

def get_minima(persistence, homology = 1):
    tmp = extract_homology(persistence, homology = homology)

    # If there is no element in the homology group, return an empty list
    if len(tmp) == 0:
        return nan
        
    # Wandle Array aus Tupel in 2D Array
    tmp_array = np.stack(tmp[:,1])

    # Finde Maximum in Persistenz
    min_value_in_persistence = np.amin(tmp_array[:,0])

    return min_value_in_persistence

def get_all_minima(all_persistences, homology = 1):
    all_minima = []
    
    for index_persistence in range(len(all_persistences)):
        one_minima = get_minima(all_persistences[index_persistence])
        
        all_minima.append(one_minima)
        
    return np.asarray(all_minima)

def pl_from_persistence(persistence, resolution = 1000, num_landscapes = 10, homology = 1, sample_range=[nan, nan]):
    
    # Filter auf eine Homologiegruppe
    persistence = extract_homology(persistence, homology = homology, remove_inf = True)
    
    if len(persistence) == 0:
        return np.full((num_landscapes*resolution), nan)
    
    # Formatierung
    persistence = np.stack(persistence[:,1], axis=0)
    
    # Aufsteigende Sortierung nach Geburts
    persistence = persistence[persistence[:, 0].argsort()]
    
    LS = gd.representations.Landscape(resolution = resolution,
                                      num_landscapes = num_landscapes,
                                      sample_range = sample_range).fit_transform([persistence])[0]
    
    return LS

def pi_from_persistence(persistence, bandwidth = 1.0, resolution = [20, 20], weight= lambda x: 1, homology = 1, im_range = [nan, nan, nan, nan]):
    
    # Filter auf eine Homologiegruppe
    persistence = extract_homology(persistence, homology = homology, remove_inf = True)
    
    if len(persistence) == 0:
        return np.full((resolution[0]*resolution[1]), nan)

    # Formatierung
    persistence = np.stack(persistence[:,1], axis=0)
    
    # Aufsteigende Sortierung nach Geburts
    persistence = persistence[persistence[:, 0].argsort()]
    
    PI = gd.representations.PersistenceImage(bandwidth = bandwidth,
                                             resolution = resolution,
                                             im_range = im_range,
                                             weight = weight).fit_transform([persistence])[0]
    
    return PI

# Hilfsfunktion zum plotten der Persistent Landscapes
def plot_pl(pl, resolution, axes = None,  num_landscapes=5, title='Landscape'):
    
    if axes is not None:
        for i in range(num_landscapes):
            axes.plot(pl[i*resolution : (i+1)*resolution])
    
    else:
        for i in range(num_landscapes):
            plt.plot(pl[i*resolution : (i+1)*resolution])
        plt.title(title)
        plt.show()
        
def plot_pi(pi, resolution = [20, 20], cmap = None, axes = None, title = "Persistence Image"):
    
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
    
    avg_pl_all_persistences = []
    
    # Extrahiere aus allen Persistenzen die jeweiligen Einträge einer Homologiegruppe
    for index_persistence in range(len(all_persistences_onesubject)):
        avg_pl_all_persistences.insert(index_persistence, pl_from_persistence(all_persistences_onesubject[index_persistence],
                                                                              resolution = resolution,
                                                                              num_landscapes = num_landscapes,
                                                                              homology = homology,
                                                                              sample_range = sample_range
                                                                             ))
    
    # Mittelwert über alle Persistenzen
    avg_pl = np.nanmean(avg_pl_all_persistences, axis=0)

    return avg_pl

def get_all_maxima(persistences_onesubject, homology = 1):
    all_maxima = []
    
    for index_persistence in range(len(persistences_onesubject)):
        tmp = extract_homology(persistences_onesubject[index_persistence], homology = homology)

        # Wandle Array aus Tupel in 2D Array
        tmp_array = np.stack(tmp[:,1])
        
        # Finde Maximum in Persistenz
        max_value_in_persistence = np.amax(tmp_array[:,1])
        
        # Füge Maximum der Liste aller Maxima hinzu
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
    print("Saving " + file)
    with open(file, 'wb') as f:
        dill.dump(x, f)

def load_file(file):
    with open(file, 'rb') as f:
        loaded_data = dill.load(f)
    
    return loaded_data

def calc_persistence_in_buckets_oneperson(t_sigma, n_subsample, n_buckets, n_per_bucket, concat_buckets=False):
    
    n_genes = shape(t_sigma)[0]
    
    # Initialisiere Listen für die Buckets
    persistence_oneperson = []
    
    for index_bucket in range(n_buckets):
        print("Berechne Bucket: ", index_bucket)
        
        # Initialisiere Listen für die Persistenzintervalle je Bucket
        persistence_onebucket = []
    
        for index_pl in range(n_per_bucket):
            
            # Subsampling
            t_sigma_short = create_subsample(t_sigma,
                                             a = n_genes,
                                             size = n_subsample)
            
            # RipsComplex
            rips_complex = gd.RipsComplex(
                distance_matrix = t_sigma_short
            ) 
            
            # Persistence
            simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
            BarCodes_Rips = simplex_tree.persistence()
            
            persistence_onebucket.insert(index_pl, BarCodes_Rips)
            
        persistence_oneperson.insert(index_bucket, persistence_onebucket)
        
    return persistence_oneperson

def calc_pers_intervals_in_buckets_oneperson(t_sigma, n_genes, n_buckets, n_per_bucket, n_subsample, concat_buckets=True ):
    
    # Initialisiere Listen für die Buckets
    pers_intervals_buckets_h0_oneperson = []
    pers_intervals_buckets_h1_oneperson = []
    
    for index_bucket in range(n_buckets):
        print("Berechne Bucket: ", index_bucket)
        
        # Initialisiere Listen für die Persistenzintervalle je Bucket
        pers_intervals_h0_onebucket = []
        pers_intervals_h1_onebucket = []
    
        for index_pl in range(n_per_bucket):
            
            # Subsampling
            t_sigma_short = create_subsample(t_sigma,
                                             a = n_genes,
                                             size = n_subsample)
            
            # RipsComplex
            rips_complex = gd.RipsComplex(
                distance_matrix = t_sigma_short
            ) 
            
            # Persistence
            simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
            BarCodes_Rips = simplex_tree.persistence()
            
            # Persistenzintervalle H0
            pers_intervals_h0 = simplex_tree.persistence_intervals_in_dimension(0)
            pers_intervals_h0 = pers_intervals_h0[pers_intervals_h0[:,1] != Inf, :]
            pers_intervals_h0_onebucket.insert(index_pl, pers_intervals_h0)
    
            # Persistenzintervalle H1
            pers_intervals_h1 = simplex_tree.persistence_intervals_in_dimension(1)
            pers_intervals_h1_onebucket.insert(index_pl, pers_intervals_h1)
        
        if concat_buckets:
            # Buckets zusammenfügen
            pers_intervals_h0_onebucket = np.concatenate(pers_intervals_h0_onebucket, axis=0)
            pers_intervals_h1_onebucket = np.concatenate(pers_intervals_h1_onebucket, axis=0)
        
        pers_intervals_buckets_h0_oneperson.insert(index_bucket, pers_intervals_h0_onebucket)
        pers_intervals_buckets_h1_oneperson.insert(index_bucket, pers_intervals_h1_onebucket)
        
    return pers_intervals_buckets_h0_oneperson, pers_intervals_buckets_h1_oneperson

def calc_avg_PL_from_all_persistences(all_persistences, homology = 1, resolution = 1000, num_landscapes = 10, scaling = "within_subjects", verbose = True):
    
    # check for between_subject scaling
    scaling_calculated = False
    
    avg_pl_allsubjects = []
    
    for index_subject in range(len(all_persistences)):
        if verbose:
            print("Calculating PL for sample no.", index_subject)
            
        avg_pl_onesubject = []
        
        # TODO Max-Extrahieren (within / between subject)
        if not scaling_calculated:
            if (scaling == "between_subjects"):
                
                #TODO get_all_maxima_allsubjects
                print("Ermittle Maximum unter Persistenzen aller Probanden")
                
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
    

# ML Funktionen ----------------------------------------------------------------------
from sklearn import metrics

def calc_accuracy(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return (TP+TN)/(TP+TN+FP+FN)

def calc_precision(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP/(TP+FP)

def calc_recall(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP/(TP+FN)

def calc_TPR(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    # Sensitivity, hit rate, recall, or true positive rate
    return TP/(TP+FN)
    
def calc_TNR(CM):
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
    
    # Modelle bereits in test_multiple_models kombiniert
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
                                  #'history': history,
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
    
    if len(models)==0:
        raise ValueError("No 'models'-input'")
        
    grid = np.array(np.meshgrid(models, models_to_combine, learning_rates, validation_splits, epochs)).T.reshape(-1,5)
    grid = pd.DataFrame(grid, columns = ['model', 'models_to_combine', 'learning_rate', 'validation_split', 'epochs'])
        
    
    return grid

def test_multiple_models(x_train, y_train, x_test, y_test, modelgrid, verbose=True):
    
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