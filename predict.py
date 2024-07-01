import cv2
from skimage.feature import hog
import mediapipe as mp
from skimage.transform import resize
import tensorflow as tf
import time
import os
import numpy as np


# Função para capturar vídeo da webcam por 10 segundos
def capture_video_with_prediction(model, label_map, duration=10, fps=30):
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Processar a imagem para HOG
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = resize(gray_frame, (640, 480))  # Redimensionar para o tamanho esperado
        features = hog(resized_frame, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features = features.reshape(1, -1)  # Reshape para uma amostra

        # Fazer a predição
        prediction = model.predict(features)[0]
        predicted_class = np.argmax(prediction)
        predicted_label = label_map[predicted_class]

        # FaceMesh processing
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Adicionar o nome na imagem
        cv2.putText(frame, predicted_label, (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_images_from_directory(directory, size=(128,128)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, size)  # Redimensiona a imagem
            images.append(resized_image)

    return np.array(images, dtype=np.float32)


# Função para carregar e redimensionar imagens
def load_and_resize_images(directory, size=(640, 480)):
    images = []
    original_images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue  # Ignora arquivos que não são imagens válidas
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_gray_image = cv2.resize(gray_image, size)  # Redimensiona a imagem em escala de cinza
            resized_color_image = cv2.resize(image, size)  # Redimensiona a imagem colorida
            images.append(resized_gray_image)
            original_images.append(resized_color_image)
    return images, original_images

# Função para extrair características HOG das imagens
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Função para classificar imagens e escrever o nome na imagem
def classify_and_label_images(model, hog_features, original_images, class_names):
    for i, features in enumerate(hog_features):
        # Adiciona dimensão para correspondência de entrada do modelo
        features = np.nan_to_num(features, nan=np.nanmean(features))
        features_expanded = np.expand_dims(features, axis=0)
        print(features_expanded)

        # Faz a predição
        prediction = model.predict(features_expanded)
        print(prediction)
        predicted_class = np.argmax(prediction)
        print(predicted_class)
        # Obtém a classe com maior probabilidade
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        # Escreve o nome na imagem original (colorida)
        cv2.putText(original_images[i], predicted_class_name, (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostra a imagem com o nome escrito
        cv2.imshow('Classified Image', original_images[i])
        cv2.waitKey(0)  # Espera pela tecla qualquer para fechar a janela

    cv2.destroyAllWindows()


# Carregar o modelo
model = tf.keras.models.load_model('./face_recognition_model.keras')
label_map = {0: 'Gabriel', 1: 'Guilherme', 2: 'Luan'}  # Mapeamento de predições para nomes

# Classificar a partir da Webcam
# capture_video_with_prediction(model, label_map)

# Classificar e rotular imagens
# Caminho do diretório
directory = './train'  # Substitua pelo caminho do seu diretório de imagens

# Carregar e redimensionar imagens
images, original_images = load_and_resize_images(directory)

# # Extrair características HOG das imagens carregadas
hog_features = extract_hog_features(images)

# Normalizar as características HOG para usar com o modelo
hog_features = (hog_features - np.mean(hog_features, axis=0)) / np.std(hog_features, axis=0)

classify_and_label_images(model, hog_features, original_images, label_map)

