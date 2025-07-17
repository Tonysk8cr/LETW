# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 10/07/2025


from VideoBatchProcessor import VideoBatchProcessor
from DataExtraction import DataExtractor
from DataLabelling import DataLabelling
from TrainingLSTM import TrainingLSTM
from RealtimePrediction import RealtimeDetection

def main():


    menu = True
    while menu:
        print("Hola, seleccione una opción:")
        print("1. Extraer datos de video")
        print("2. Procesar videos en lote")
        print("3. Label Data")
        print("4. Train LSTM")
        print("5. Detección en tiempo real")
        print("6. Salir \n")
        user_choice = input("Ingrese su opción (1/2/3/4/5/6): ")
        if user_choice == '1':
            print("Extracción de datos de video: " \
            "\ Opcinones: \n1. Extraer datos de un video específico " \
            "\n2. Procesar todos los videos en un directorio" \
            "\n3. Regresar \n")
            user_choice = input("Seleccione una opción: ")

            if user_choice == '1':
                print("Extrayendo datos de un video específico...")
                video_path = r"C:\Users\tonyi\LETW\Model\Vids"
                processor = VideoBatchProcessor(directory=video_path)
                processor.extract_single_path()
            elif user_choice == '2':
                print("Extrayendo datos de todos los videos de un directorio padre")
                parent_directory = r"C:\Users\tonyi\LETW\Model\Test\Test_Videos"
                processor = VideoBatchProcessor(directory=parent_directory)
                processor.extract_parent_path()
            elif user_choice == '3':
                print("Regresando al menú principal...\n")
                continue
            else:
                print("Opción no válida. Por favor, intente de nuevo. \n")
                continue

        elif user_choice == '2':
            print("Opciones de Procesamiento de videos: ")
            print("Ojo solo para revisar los videos, no para extraer datos")
            print("1. Extraer datos de un video específico")
            print("2. Procesar todos los videos en un directorio")
            print("3. Regresar")
            user_choice2 = input("Seleccione una opción: ")

            if user_choice2 == '1':
                video_path = r"C:\Users\tonyi\LETW\Model\Vids"
                processor = VideoBatchProcessor(directory=video_path)
                processor.run()
            elif user_choice2 == '2':
                videos_directory = r"C:\Users\tonyi\LETW\Model\Test\Test_Videos"
                processor = VideoBatchProcessor(videos_directory)
                processor.train()
            elif user_choice2 == '3':
                print("Regresando al menú principal... \n")
                continue
            else:
                print("Opción no válida. Por favor, intente de nuevo. \n")
                continue

        elif user_choice == '3':
            labeller = DataLabelling()
            labeller.split_data()

        elif user_choice == '4':
            training = TrainingLSTM()
            training.build_model()

        elif user_choice == '5':
            print("Prueba de deteccion: ")
            deteccion = RealtimeDetection()
            deteccion.real_time_detection()

        elif user_choice == '6':
            print("Saliendo del programa. ¡Hasta luego!")
            menu = False
            return

        else:
            print("Opción no válida. Por favor, intente de nuevo. \n")



if __name__ == '__main__':
    main()