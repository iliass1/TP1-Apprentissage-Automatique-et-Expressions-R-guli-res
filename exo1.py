# Importation des bibliothèques nécessaires pour le projet CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import re

# =====================
# EXERCICE 1 : CLASSIFICATION D'IMAGES AVEC CNN
# =====================

# Étape 1 : Préparation de l'environnement (pas nécessaire ici car les bibliothèques sont importées)

# Étape 2 : Chargement du dataset et normalisation
# Charger le dataset Fashion-MNIST directement depuis Keras

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalisation des images (les valeurs des pixels doivent être entre 0 et 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Affichage d'une image et de son étiquette pour mieux comprendre le dataset
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Étiquette : {y_train[0]}')  # Afficher l'étiquette associée
plt.show()

# Étape 3 : Création du modèle CNN
# Reshape des images pour qu'elles aient une forme compatible avec les réseaux CNN (ajout de la dimension des canaux)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Création d'un modèle CNN basique
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Première couche de convolution
    MaxPooling2D(2, 2),  # Couche de pooling pour réduire la taille de l'image
    Conv2D(64, (3, 3), activation='relu'),  # Deuxième couche de convolution
    MaxPooling2D(2, 2),  # Deuxième couche de pooling
    Flatten(),  # Aplatir la matrice en un vecteur
    Dense(128, activation='relu'),  # Couche dense (fully connected)
    Dense(10, activation='softmax')  # Couche de sortie pour les 10 classes avec activation softmax
])

# Compilation du modèle, on définit ici l'optimiseur, la fonction de perte, et les métriques
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Étape 4 : Entraînement du modèle
# Entraîner le modèle sur les données d'entraînement (x_train, y_train)
model.fit(x_train, y_train, epochs=10, batch_size=64)

# Évaluation du modèle sur les données de test pour voir la précision
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision sur les données de test : {test_acc}")

# Étape 5 : Prédiction sur les données de test
# Utilisation du modèle pour prédire les classes des images du test
predictions = model.predict(x_test)

# Fonction pour afficher une image, sa prédiction et l'étiquette réelle
def afficher_resultat(index):
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f'Prédiction : {np.argmax(predictions[index])}, Étiquette réelle : {y_test[index]}')
    plt.show()

# Afficher un exemple de prédiction
afficher_resultat(0)


# =====================
# EXERCICE 2 : EXPRESSIONS RÉGULIÈRES AVANCÉES
# =====================

# Exemple de texte contenant des emails, numéros de téléphone, dates et URLs
texte = """
Voici un exemple de texte avec différentes informations :
- Email : john.doe@example.com, jane_doe123@site.fr
- Téléphone : +33 6 12 34 56 78, (555) 123-4567, 1234567890
- Dates : 12/08/2021, 08-15-2022, 01/01/23
- URLs : https://example.com, http://site.com
"""

# 1. Extraction des adresses email valides
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', texte)
print(f"Emails extraits : {emails}")

# 2. Extraction des numéros de téléphone dans différents formats
telephones = re.findall(r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', texte)
print(f"Numéros de téléphone extraits : {telephones}")

# 3. Extraction des dates dans plusieurs formats (DD/MM/YYYY, MM-DD-YYYY)
dates = re.findall(r'\b\d{2}[/-]\d{2}[/-]\d{2,4}\b', texte)
print(f"Dates extraites : {dates}")

# 4. Extraction des URLs et vérification si elles utilisent le protocole HTTPS
urls = re.findall(r'https://[a-zA-Z0-9./?=_-]+', texte)
print(f"URLs extraites (HTTPS uniquement) : {urls}")

