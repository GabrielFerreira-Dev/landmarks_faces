import cv2
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_tuner.tuners import RandomSearch
import time
from sklearn.model_selection import train_test_split
import mediapipe as mp


# Função para capturar vídeo da webcam por 10 segundos
def capture_video_from_webcam(duration=10, fps=30):
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    cap = cv2.VideoCapture(0)
    frames = []
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
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


# Capturar vídeo da webcam
frames = capture_video_from_webcam()


# Função para extrair características HOG das imagens
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)


# Extrair características HOG dos frames capturados
hog_features = extract_hog_features(frames)
labels = np.zeros(len(hog_features))  # Rótulo genérico, ajustável conforme necessidade

X_train, X_val, y_train, y_val = train_test_split(hog_features, labels, test_size= 0.2, random_state= 42)

# Configurar a rede neural
model = Sequential([
    Dense(128, activation='relu', input_shape=(hog_features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Como temos um único rótulo, usamos a ativação sigmoid
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar a rede neural
model.fit(hog_features, labels, epochs=5, batch_size=1, validation_split=0.0)


# Função para construir o modelo com hiperparametrização
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(hog_features.shape[1],)))
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
tuner.search(X_train, y_train, epochs=5, validation_data= (X_val, y_val))

# Obter o melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# Salvar o modelo treinado
best_model.save('face_recognition_model.keras')
