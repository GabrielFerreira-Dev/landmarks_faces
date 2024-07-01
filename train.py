import cv2
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras_tuner.tuners import RandomSearch
import mediapipe as mp
import os


# Função para carregar imagens de um diretório e convertê-las para escala de cinza
def load_images_from_directory(directory, size=(640,480)):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, size)  # Redimensiona a imagem
            images.append(resized_image)

            if 'Gabriel' in filename:
                labels.append(0)  # Label para outros casos
            elif 'Guilherme' in filename:
                labels.append(1)  # Label para outros casos
            elif 'Luan' in filename:
                labels.append(2)
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    return images, labels

# Função para extrair características HOG das imagens
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Carregar imagens de um diretório
directory = './Images'  # Substitua pelo caminho do seu diretório de imagens
images, labels = load_images_from_directory(directory)

# Extrair características HOG das imagens carregadas
hog_features = extract_hog_features(images)

num_classes = 3

# Configurar a rede neural
model = Sequential([
    Input(shape=(hog_features.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Como temos um único rótulo, usamos a ativação sigmoid
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar a rede neural
model.fit(hog_features, labels, epochs=10, batch_size=1, validation_split=0.0)


# Função para construir o modelo com hiperparametrização
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(hog_features.shape[1],)))
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


# Configurar o Keras Tuner para hiperparametrização
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='my_dir',
                     project_name='facial_recognition', )

# Realizar a busca de hiperparâmetros
tuner.search(hog_features, labels, epochs=10, validation_split=0.2)  # Usar 20% dos dados para validação
print('Passou')

# Obter o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]

print(best_model.summary())

# Salvar o modelo treinado
best_model.save('face_recognition_model.keras')
