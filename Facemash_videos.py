import cv2
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from kerastuner.tuners import RandomSearch
import mediapipe as mp
import time
import os


# Função para capturar vídeo de arquivos em um diretório
def capture_video_from_directory(directory="Videos", duration=1, fps=30):
    video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp4', '.avi'))]
    frames = []
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
    target_size = (640, 480)  # Tamanho fixo da janela

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        start_time = time.time()
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            # Redimensionar o frame para um tamanho fixo
            frame = cv2.resize(frame, target_size)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)

            # FaceMesh processing
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            cv2.imshow('frame', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()
    return frames


# Capturar vídeo dos arquivos no diretório
frames = capture_video_from_directory()


# Função para extrair características HOG das imagens
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features, dtype=np.float32)  # Garantir que o tipo de dados seja float32


# Extrair características HOG dos frames capturados
hog_features = extract_hog_features(frames)
labels = np.zeros(len(hog_features), dtype=np.float32)  # Rótulo genérico, ajustável conforme necessidade

# Configurar a rede neural
model = Sequential([
    Input(shape=(hog_features.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Como temos um único rótulo, usamos a ativação sigmoid
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar a rede neural
model.fit(hog_features, labels, epochs=20, batch_size=1, validation_split=0.0)


# Função para construir o modelo com hiperparametrização
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(hog_features.shape[1],)))
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


# Configurar o Keras Tuner para hiperparametrização
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='my_dir',
                     project_name='facial_recognition')

# Realizar a busca de hiperparâmetros
tuner.search(hog_features, labels, epochs=20, validation_split=0.2)  # Usar 20% dos dados para validação

# Obter o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# Salvar o modelo treinado
best_model.save('face_recognition_model.h5')
