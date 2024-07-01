import cv2
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras_tuner.tuners import RandomSearch
import mediapipe as mp
import os


def extract_hog_from_face_region(frame):
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

    # Convertendo a imagem de BGR para RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processando a imagem com o modelo FaceMesh
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        # Supondo que queremos apenas o primeiro rosto detectado
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extraindo coordenadas dos pontos-chave faciais (neste caso, apenas a região do rosto)
        face_points = []
        for landmark in face_landmarks.landmark:
            # Convertendo para coordenadas da imagem
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            face_points.append((x, y))
        
        # Definindo a região do rosto com base nos pontos-chave
        if len(face_points) >= 7:  # Verifica se há pontos suficientes para definir a região do rosto
            # Selecionando os pontos do contorno da face
            face_roi = np.array(face_points[1:7])  # Aqui estamos pegando os pontos do contorno do rosto
            
            # Garantindo que as coordenadas sejam números inteiros para indexação
            face_roi = face_roi.astype(int)
            
            # Limites para recorte da região do rosto
            top = np.min(face_roi[:, 1])
            bottom = np.max(face_roi[:, 1])
            left = np.min(face_roi[:, 0])
            right = np.max(face_roi[:, 0])
            
            # Recorte da região do rosto da imagem original (frame)
            face_image = frame[top:bottom, left:right]
            
            # Verifica se a imagem recortada é grande o suficiente para o HOG
            if face_image.shape[0] >= 16 and face_image.shape[1] >= 16:
                # Convertendo a imagem para escala de cinza
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (64, 128))
                hog_features = hog(gray_face, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                
                
                return hog_features
            else:
                print("A região do rosto recortada é muito pequena para calcular HOG.")
                return None
        else:
            print("Número insuficiente de pontos faciais detectados para definir a região do rosto.")
            return None
    else:
        print("Nenhum rosto detectado na imagem.")
        return None


def capture_images_from_directory(directory="Images"):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    frames = []
    labels = []

    for image_file in image_files:        
        frame = cv2.imread(image_file)
        if frame is None:
            continue
        
        # Extrair características HOG da região do rosto
        hog_features = extract_hog_from_face_region(frame)
        
        if hog_features is not None:
            frames.append(hog_features)
            # Definir label baseado no nome do arquivo (exemplo genérico)
            if 'Luan' in image_file:
                labels.append(0)
            elif 'Gabriel' in image_file:
                labels.append(1)  # Label para outros casos
            elif 'Guilherme' in image_file:
                labels.append(2)  # Label para outros casos
            

    # Verificar se frames e labels têm o mesmo comprimento
    if len(frames) != len(labels):
        print("Número de frames e labels não corresponde.")
        return None, None
        
    frames = np.array(frames, dtype=np.float32)
    labels = np.array(labels)
    
    return frames, labels


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

            if 'Luan' in filename:
                labels.append(0)
            elif 'Gabriel' in filename:
                labels.append(1)  # Label para outros casos
            elif 'Guilherme' in filename:
                labels.append(2)  # Label para outros casos
    
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


# Capturar vídeo dos arquivos no diretório
# frames = capture_video_from_directory()
# hog_features, labels = capture_images_from_directory()

print(hog_features.shape, labels.shape)
print(labels)
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
model.fit(hog_features, labels, epochs=20, batch_size=1, validation_split=0.0)


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
tuner.search(hog_features, labels, epochs=20, validation_split=0.2)  # Usar 20% dos dados para validação
print('Passou')

# Obter o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]

print(best_model.summary())

# Salvar o modelo treinado
best_model.save('face_recognition_model.keras')
