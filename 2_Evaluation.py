#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================
import os
from PIL import Image

# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model


# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modèle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# L'emplacement des images de test
mainDataPath = "donnees/"
testPath = mainDataPath + "test"

# Le nombre des images(1500 images pour la classe du dauphin et 1500 pour la classe du requin)
number_images = 3000
number_images_class_0 = 1500
number_images_class_1 = 1500

# La taille des images
image_scale = 80

# La couleur des images(grayscale or rgb)
images_color_mode = "rgb"

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="binary",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images

(x, y_true) = test_itr.next()

# Normalize Data
max_value = float(x.max())
x = x.astype('float32') / max_value

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

# evaluation du modèle
#test_eval = Classifier.evaluate_generator(test_itr, verbose=1)     #ICI
test_eval = Classifier.evaluate(test_itr, verbose=1)                #AJOUT

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
#predicted_classes = Classifier.predict_generator(test_itr, verbose=1)  #ICI
predicted_classes = Classifier.predict(test_itr, verbose=1)             #AJOUT
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.round(predicted_classes) # on arrondie le output
# 0 => classe dauphin
# 1 => classe requin

# Cette liste contient les images bien classées
correct = []
for i in range(0, len(predicted_classes) - 1):
    if predicted_classes[i] == y_true[i]:
        correct.append(i)

# Nombre d'images bien classées
print("> %d  étiquettes bien classées" % len(correct))

# Cette list contient les images mal classées
incorrect = []
for i in range(0, len(predicted_classes) - 1):
    if predicted_classes[i] != y_true[i]:
        incorrect.append(i)

# Nombre d'images mal classées
print("> %d étiquettes mal classées" % len(incorrect))


# 2) Afficher la matrice de confusion

cm = confusion_matrix(y_true, predicted_classes)
print("Confusion Matrix:")
print(cm)

# 3) Afficher la courbe ROC

fpr, tpr, thresholds = roc_curve(y_true, predicted_classes_perc)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 4) Extraire 5 images de chaque catégorie

# ***************************************************************
#  les images bien classées (dauphins et requins)
# ***************************************************************

# Fonction pour afficher les images des dauphins et requins bien classées
# Elle prend en paramètre: 
#   1) La liste des images bien classées,
#   2) Une liste des étiquettes des 2 classes (Dauphin et Requin),
#   3) Une valeur indiquant la taille à laquelle les images doivent être redimensionnées avant l'affichage.

def display_images_bien_classe(correct, class_labels, image_scale):
    fig, ax = plt.subplots(2, 5, figsize=(15, 7))   # Afficher 2 rangées d'images, chacune contenant cinq images.
    dauphin_count = 0
    requin_count = 0

    for i in range(len(correct)):
        if dauphin_count == 5 and requin_count == 5:
            break

        image_index = correct[i]
        true_class = class_labels[y_true[image_index]]  # Vérifier la vraie classe de l'image
        if true_class == "dauphin" and dauphin_count < 5:
            row = 0
            col = dauphin_count
            dauphin_count += 1
        elif true_class == "requin" and requin_count < 5:
            row = 1
            col = requin_count
            requin_count += 1
        else:
            continue
        
        # Charger l'image à partir de son chemin d'accès complet (test_itr.filepaths[image_index])
        # et redimensionner à la taille spécifiée par image_scale.
        image_path = os.path.join(test_itr.filepaths[image_index])
        image = Image.open(image_path)
        image = image.resize((image_scale, image_scale))
        # L'image est affichée sur la sous-figure correspondante dans la figure matplotlib ax[row, col].
        # Désactiver l'affichage des axes (bords) autour de l'image, (Avec axis('off')).
        ax[row, col].imshow(image)
        ax[row, col].axis('off')

    # Ajout des annotations Dauphins Bien Classés et Requins Bien Classés aux figures.
    ax[0, 0].annotate("Dauphins Bien Classés", xy=(0, 0), xytext=(5, 5),
                      textcoords="offset points", color="white", fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="green"))
    ax[1, 0].annotate("Requins Bien Classés", xy=(0, 0), xytext=(5, 5),
                      textcoords="offset points", color="white", fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="blue"))

    # Si le nombre de dauphins correctement classés est inférieur à cinq,
    # les images restantes dans la première rangée sont désactivées.
    for i in range(dauphin_count, 5):
        ax[0, col].axis('off')
    # Si le nombre de requins correctement classés est inférieur à cinq, 
    # les images restantes dans la deuxième rangée sont aussi désactivées
    for i in range(requin_count, 5):
        ax[1, col].axis('off')

    plt.show()

# ***************************************************************
#  les images mal classées (dauphins et requins)
# ***************************************************************

# Fonction pour afficher les images des dauphins et requins mal classées
# Elle prend en paramètre: 
#   1) La liste des images mal classées,
#   2) Une liste des étiquettes des 2 classes (Dauphin et Requin),
#   3) Une valeur indiquant la taille à laquelle les images doivent être redimensionnées avant l'affichage.

def display_images_mal_classe(incorrect, class_labels, image_scale):
    fig, ax = plt.subplots(2, 5, figsize=(15, 7))
    dauphin_count = 0
    requin_count = 0

    for i in range(len(incorrect)):
        if dauphin_count == 5 and requin_count == 5:
            break

        image_index = incorrect[i]
        true_class = class_labels[y_true[image_index]]
        predicted_class = class_labels[int(predicted_classes[image_index])]
        if true_class == "dauphin" and predicted_class == "requin" and dauphin_count < 5:
            row = 0
            col = dauphin_count
            dauphin_count += 1

        elif true_class == "requin" and predicted_class == "dauphin" and requin_count < 5:
            row = 1
            col = requin_count
            requin_count += 1

        else:
            continue

        image_path = os.path.join(test_itr.filepaths[image_index])
        image = Image.open(image_path)
        image = image.resize((image_scale, image_scale))

        ax[row, col].imshow(image)
        ax[row, col].axis('off')

    ax[0, 0].annotate("Dauphins classés comme Requins", xy=(0, 0), xytext=(5, 5),
                      textcoords="offset points", color="white", fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="green"))
    ax[1, 0].annotate("Requins classés comme Dauphins", xy=(0, 0), xytext=(5, 5),
                      textcoords="offset points", color="white", fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="blue"))

    for i in range(dauphin_count, 5):
        ax[0, col].axis('off')

    for i in range(requin_count, 5):
        ax[1, col].axis('off')

    plt.show()


# appeler des méthodes pour afficher les images 
display_images_bien_classe(correct, {0: "dauphin", 1: "requin"}, image_scale)
display_images_mal_classe(incorrect, {0: "dauphin", 1: "requin"},image_scale)