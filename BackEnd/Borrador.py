# Developed by Anthony Villalobos 08/01/2025
# Adaptado para usar un VIDEO en lugar de la cámara
import cv2
import time
import mediapipe as mp
import numpy as np
import os
import random
from matplotlib import pyplot as plt

"""
Esta clase se encarga de procesar un video para hacer la detección de:
 - Rostro
 - Manos (izquierda y derecha)
 - Pose (cuerpo)
y extraer sus keypoints.
"""
class ProcesamientoImagen:
    """
    Clase para procesar el video y extraer los keypoints de los landmarks detectados.
    """
    def __init__(self):
        # Inicializamos componentes de MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(self, frame, model):
        """
        Aplica el modelo Mediapipe (Holistic) para obtener detecciones del frame.
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = model.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Regresar la imagen a BGR para dibujar y mostrar con OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr, results

    def draw_styled_landmarks(self, frame, results):
        """
        Dibuja los landmarks (rostro, pose, manos) detectados.
        """
        # Rostro
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(103,207,245), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
            )
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )
        # Mano izquierda
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=3)
            )
        # Mano derecha
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=3)
            )

    def image_processing_from_video(self, video_path):
        """
        Función para abrir un archivo de video y realizar la detección de landmarks.
        Retorna el último frame con detecciones y sus resultados correspondientes.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")

            print(f"Iniciando procesamiento del video: {video_path}")
            print("Presione 'q' para salir manualmente en cualquier momento...")

            last_valid_result = None
            last_valid_frame = None

            with self.mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as holistic:

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        # No hay más frames o error al leer
                        print("Fin del video o problema al leer el frame.")
                        break

                    # Detecciones con MediaPipe
                    image, results = self.mediapipe_detection(frame, holistic)

                    # Guardar el último frame con detecciones si se detecta algo
                    if (results.pose_landmarks or
                        results.face_landmarks or
                        results.left_hand_landmarks or
                        results.right_hand_landmarks):

                        last_valid_result = results
                        last_valid_frame = frame

                        # Mensaje de consola
                        print("Detectado: ", end="")
                        if results.pose_landmarks:
                            print("Pose ", end="")
                        if results.face_landmarks:
                            print("Cara ", end="")
                        if results.left_hand_landmarks:
                            print("Mano izq ", end="")
                        if results.right_hand_landmarks:
                            print("Mano der ", end="")
                        print("")  # Salto de línea

                    # Agregar texto de estado en la ventana
                    status_text = "Estado:"
                    if results.pose_landmarks:
                        status_text += " Pose"
                    if results.face_landmarks:
                        status_text += " + Cara"
                    if results.left_hand_landmarks:
                        status_text += " + Mano Izq"
                    if results.right_hand_landmarks:
                        status_text += " + Mano Der"

                    cv2.putText(
                        image, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2, cv2.LINE_AA
                    )

                    # Dibujar landmarks en la imagen
                    self.draw_styled_landmarks(image, results)

                    # Mostrar el frame procesado
                    cv2.imshow('Detección desde video', image)


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Saliendo por tecla 'q'...")
                        break

            cap.release()
            cv2.destroyAllWindows()
            return last_valid_frame, last_valid_result

        except Exception as e:
            print(f"Error al procesar video: {e}")
            return None, None

    def last_frame(self, frame):
        """
        Muestra el último frame en una ventana de matplotlib.
        """
        if frame is not None:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()
            return True
        return False

    def extract_keypoints(self, results):
        """
        Extrae los keypoints (pose, cara, manos izq y der) y los concatena en un vector.
        """
        try:
            # Pose: 33 puntos * 4 valores = 132 (x, y, z, visibility)
            pose = (np.array([[res.x, res.y, res.z, res.visibility] 
                              for res in results.pose_landmarks.landmark]).flatten()
                    if results.pose_landmarks else np.zeros(33*4))

            # Cara: 468 puntos * 3 valores = 1404 (x, y, z)
            face = (np.array([[res.x, res.y, res.z]
                              for res in results.face_landmarks.landmark]).flatten()
                    if results.face_landmarks else np.zeros(468*3))

            # Mano izquierda: 21 puntos * 3 = 63
            lh = (np.array([[res.x, res.y, res.z]
                            for res in results.left_hand_landmarks.landmark]).flatten()
                  if results.left_hand_landmarks else np.zeros(21*3))

            # Mano derecha: 21 puntos * 3 = 63
            rh = (np.array([[res.x, res.y, res.z]
                            for res in results.right_hand_landmarks.landmark]).flatten()
                  if results.right_hand_landmarks else np.zeros(21*3))

            # Concatenar
            keypoints = np.concatenate([pose, face, lh, rh])
            return keypoints, True
        except Exception as e:
            print(f"Error al extraer los keypoints: {e}")
            return None, False

def main():
    # Instancia de la clase
    procesador = ProcesamientoImagen()

    # Ruta al video que deseas procesar
    directorio_videos = r"C:\Users\tonyi\OneDrive\Documentos\LETW\Test_Videos"

    #COnjtadores para medir el tiempo de procesamiento
    contador = 0
    inicio_contador = time.perf_counter()
    extensiones_validas = ('.mp4', '.avi', '.mov', '.mkv')
    # Recorremos los archivos en el directorio
    try:
        for archivo in os.listdir(directorio_videos):
            if archivo.lower().endswith(extensiones_validas):
                ruta_video = os.path.join(directorio_videos, archivo)
                contador += 1
                print(f"Procesando video: {ruta_video}")

                # Procesar el video y obtener el último frame + resultados
                frame, results = procesador.image_processing_from_video(ruta_video)

                # Si hubo detecciones, extraer keypoints
                if results is not None:
                    keypoints, success = procesador.extract_keypoints(results)
                    if success:
                        print(f"Keypoints extraídos correctamente. Tamaño: {len(keypoints)}")
                    else:
                        print("No se pudieron extraer los keypoints.")
                else:
                    print("No se obtuvieron resultados del video.")
    finally:
        fin_contador = time.perf_counter()
        duracion = fin_contador - inicio_contador
        print(f"Tiempo total de procesamiento: {duracion:.2f} segundos")
        print(f"Total de videos procesados: {contador}")


if __name__ == '__main__':
    main()
