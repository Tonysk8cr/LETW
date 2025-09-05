# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 05/09/2025

from VideoBatchProcessor import VideoBatchProcessor
from DataLabelling import DataLabelling
from TrainingLSTM import TrainingLSTM
from RealtimePrediction import RealtimeDetection
from SetUp import SetUp

def main():

    repetitions = 100
    frames = 30
    signs = ["HOLA", "ADIÓS", "POR-FAVOR", "GRACIAS", "SI", "NO", "BIEN", "MAL", "MAMÁ", "PAPÁ"]

    menu = True
    while menu:
        print("Hola, seleccione una opción:")
        print("1. Crear directorios necesarios")
        print("2. Procesar y extraer datos de video")
        print("3. Procesar videos en lote")
        print("4. Label Data")
        print("5. Train LSTM")
        print("6. Detección en tiempo real")
        print("7. Salir \n")
        user_choice = input("Ingrese su opción (1/2/3/4/5/6/7): ")
        
        if user_choice == '1':
            print("Creando directorios necesarios...")
            setup = SetUp(repetitions, signs=signs)
            Data_Path, actions, video_path = setup.create_directories()
            print(f"Directorios creados en {Data_Path} para las acciones: {actions}")

        elif user_choice == '2':
            print("Extracción de datos de video: " \
            "Opciones: " \
            "\n1. Extraer datos de un video específico " \
            "\n2. Procesar todos los videos en un directorio" \
            "\n3. Regresar \n")
            user_choice = input("Seleccione una opción: ")

            if user_choice == '1':
                print("Extrayendo datos de un video específico...")
                video_path = "./Model/Vids"
                processor = VideoBatchProcessor(directory=video_path, repetitions=repetitions, signs=signs, frames=frames)
                processor.extract_single_path()
            elif user_choice == '2':
                print("Extrayendo datos de todos los videos de un directorio padre")
                parent_directory = "./Model/Test/Test_Videos"
                processor = VideoBatchProcessor(directory=parent_directory, repetitions=repetitions, signs=signs, frames=frames)
                processor.extract_parent_path()
            
            elif user_choice == '3':
                print("Regresando al menú principal...\n")
                continue
            else:
                print("Opción no válida. Por favor, intente de nuevo. \n")
                continue

        elif user_choice == '3':
            print("Opciones de Procesamiento de videos: ")
            print("Ojo solo para revisar los videos, no para extraer datos")
            print("1. Extraer datos de un video específico")
            print("2. Procesar todos los videos en un directorio")
            print("3. Regresar")
            user_choice2 = input("Seleccione una opción: ")

            if user_choice2 == '1':
                video_path = "./T_Videos/"
                processor = VideoBatchProcessor(directory=video_path, repetitions=repetitions)
                processor.run()
            elif user_choice2 == '2':
                videos_directory = "./Model/Test/Test_Videos"
                processor = VideoBatchProcessor(videos_directory, repetitions=repetitions)
                processor.train()
            elif user_choice2 == '3':
                print("Regresando al menú principal... \n")
                continue
            else:
                print("Opción no válida. Por favor, intente de nuevo. \n")
                continue

        elif user_choice == '4':
            labeller = DataLabelling(repetitions=repetitions, signs=signs, frames=frames)
            labeller.split_data()

        elif user_choice == '5':
            training = TrainingLSTM(signs=signs, repetitions=repetitions, frames=frames)
            training.build_model()

        elif user_choice == '6':
            print("Prueba de deteccion: ")
            deteccion = RealtimeDetection(signs=signs)
            deteccion.real_time_detection()

        elif user_choice == '7':
            print("Saliendo del programa. ¡Hasta luego!")
            menu = False
            return

        else:
            print("Opción no válida. Por favor, intente de nuevo. \n")



if __name__ == '__main__':
    main()