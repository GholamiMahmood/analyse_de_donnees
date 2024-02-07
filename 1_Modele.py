# Ce modèle est un classifieur (un CNN) entrainé sur un ensemble de données afin 
# de distinguer entre les images de dauphins et de requins.
#
# Données:
# ------------------------------------------------
# entrainement : classe 'dauphins': 5 500 images | classe 'requins': 5 500 images
# validation   : classe 'dauphins': 1 000 images | classe 'requins': 1 000 images
# test         : classe 'dauphins': 1 500 images | classe 'requins': 1 500 images 
# ------------------------------------------------

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire

from keras.preprocessing.image import ImageDataGenerator

# Le Type de notre modèle (séquentiel)

from keras.models import Model
from keras.models import Sequential

# Le type d'optimisateur utilisé dans notre modèle est: adam.
# L'optimisateur ajuste les poids de notre modèle par descente du gradient

from keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation, Dropout, Flatten, Dense

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler

# Configuration du GPU
import tensorflow as tf

# Affichage des graphes 
import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np

# ==========================================
# ===============Début du timer ============
# ==========================================
import time
start_time = time.time()

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajustement des variables suivantes selon votre problème:
# - mainDataPath
# - training_batch_size
# - validation_batch_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le dossier contenant les images de test
testPath = mainDataPath + "test"

# Le nom du fichier du modèle à sauvegarder
modelsPath = "Model.hdf5"

# Le nombre d'images d'entrainement et de validation

training_batch_size = 11000  # total 11000 (5500 classe: dauphins et 5500 classe: requins)
validation_batch_size = 2000  # total 2000 (1000 classe: dauphins et 1000 classe: requins)

# Configuration des  images
image_scale = 80    # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs 
image_shape = (image_scale, image_scale, image_channels) # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 64 # le nombre d'images entrainées ensemble: un batch
fit_epochs = 50     # Le nombre d'époques 

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) feature_extraction
# 3) fully_connected
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)

# Partie feature extraction (ou cascade de couches d'extraction des caractéristiques)
# Cette fonction implémente une architecture de réseau de neurones avec plusieurs couches de convolutions, 
# suivies de normalisations par lot (Batch Normalization), d'activations ReLU et de couches de max pooling.
def feature_extraction(input):
  
    # 1- Couche de convolution avec nombre de filtre: 64, la taille de la fenetre de ballaiage: 3x3,
    #    La padding='same' pour que la sortie ait la même taille que l'entrée.
    # 2- Normalisation sur la sortie de la couche de convolution précédente.
    # 3- Fonction d'activation: relu, sur la sortie de la couche de Batch Normalization.
    # 4- couche d'echantillonage (max pooling) pour reduire la taille avec la taille de la fenetre de ballaiage: 2x2  
    
    # **** On répète ces étapes tant que nécessaire ****
    # À chaque étape on va augmenter le nombre de filtre
    
    x = Conv2D(64, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # 1- On répète la même étape, cette fois-ci en augmentant le nombre de filtre à 128,
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # 1- On répète la même étape, cette fois-ci en augmentant le nombre de filtre à 256,
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # 1- On répète la même étape, cette fois-ci en augmentant le nombre de filtre à 512,
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # L'ensemble des features/caractéristiques extraits
    
    return encoded

# Partie complétement connectée (Fully Connected Layer)
def fully_connected(encoded):
    # Flatten: pour convertir les matrices en vecteurs pour la couche MLP
    # Dense: une couche neuronale simple avec le nombre de neurones = 512
    # fonction d'activation: relu
    x = Flatten()(encoded)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dropout(0.4)(x)  # Couche Dropout avec un taux de 0.4

    # La dernière couche est formée d'un seul neurone avec une fonction d'activation sigmoide
    # La fonction sigmoide nous donne une valeur entre 0 et 1
    # On considère les résultats <=0.5 comme l'image appartenant à la classe 0 (c.-à-d. la classe qui correspond au dauphin)
    # on considère les résultats >0.5 comme l'image appartenant à la classe 1 (c.-à-d. la classe qui correspond au requin)
    x = Dense(1)(x)
    sortie = Activation('sigmoid')(x)
    return sortie

# Déclaration du modèle:
# La sortie de l'extracteur des features sert comme entrée à la couche complétement connectée
model = Model(input_layer, fully_connected(feature_extraction(input_layer)))

# Affichage des paramètres du modèle
# Cette commande affiche un tableau avec les détails du modèle 
# (nombre de couches et de paramétrer ...)
model.summary()

# Compilation du modèle :
# On définit la fonction de perte (exemple :loss='binary_crossentropy' ou loss='mse')
# L'optimisateur utilisé avec ses paramètres (Exemple : optimizer=adam(learning_rate=0.001) )
# La valeur à afficher durant l'entrainement, metrics=['accuracy']
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# quand il charge les images, il les ajuste (change la taille, les dimensions, la direction ...) 
# aléatoirement afin de rendre le modèle plus robuste à la position du sujet dans les images
# Note: On peut utiliser cette méthode pour augmenter le nombre d'images d'entrainement (data augmentation)
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,)

# validation_data_generator: charge les données de validation en memoire
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale), # taille des images
    batch_size=training_batch_size, # nombre d'images à entrainer (batch size)
    class_mode="binary", # classement binaire (problème de 2 classes)
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# validation_generator: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    validationPath, # Place des images de validation
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_batch_size,  # nombre d'images à valider
    class_mode="binary",  # classement binaire (problème de 2 classes)
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# On imprime l'indice de chaque classe (Keras numerote les classes selon l'ordre des dossiers des classes)
# Dans ce cas => [dauphin: 0 et requin:1]
print(training_generator.class_indices)
print(validation_generator.class_indices)

# On charge les données d'entrainement et de validation
# x_train: Les données d'entrainement
# y_train: Les étiquettes des données d'entrainement
# x_val: Les données de validation
# y_val: Les étiquettes des données de validation
(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()

# On Normalise les images en les divisant par la plus grande pixel dans les images (generalement c'est 255)
# Alors on aura des valeur entre 0 et 1, ceci stabilise l'entrainement
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec la meilleure validation accuracy ('val_acc') 
# Note: on sauvegarder le modèle seulement quand la précision de la validation s'améliore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# Fonction qui ajuste le taux d'apprentissage en fonction de l'epoch afin d'aider le modèle à mieux converger.
# Elle prend en argument une epoch et renvoie un taux d'apprentissage associé à cette epoch.
# Le taux d'apprentissage commence à 1e-3 et est réduit par un facteur de 10 aux époques 11, 21 et 31, 
# en le multipliant par des facteurs (0,1, 0,01 et 0,001)
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    elif epoch > 30:
        lr *= 0.001
    return lr

# Objet créé en utilisant LearningRateScheduler avec la fonction lr_schedule
lr_scheduler = LearningRateScheduler(lr_schedule)

# entrainement du modèle
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs, # nombre d'époques
                       batch_size=fit_batch_size, # nombre d'images entrainées ensemble
                       validation_data=(x_val, y_val), # données de validation
                       verbose=1,   # mets cette valeur à 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint, lr_scheduler],
                       shuffle=True)    # shuffle les images

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
#
# ***********************************************

# Plot accuracy over epochs (precision par époque)
print(classifier.history.keys())
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Ajouter la courbe de perte (loss curve)
#
# ***********************************************
# 
# La courbe d’erreur par époque (Training vs Validation)
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()

# ==========================================
# ===============Fin du timer =============
# ==========================================
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")