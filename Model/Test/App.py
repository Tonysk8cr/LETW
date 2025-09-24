# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
# Updated by Anthony Villalobos 23/09/2025

from sklearn import logger
from VideoBatchProcessor import VideoBatchProcessor
from DataLabelling import DataLabelling
from TrainingLSTM import TrainingLSTM
from RealtimePrediction import RealtimeDetection
from SetUp import SetUp
from Utilities import Utilities




#Options for the main menu
#---------------------------------

#The confidence variables is asked inside the options that require it
#This helps the developer to test different confidence values without having to restart the program

def option1(repetitions, signs, logger):
    print("Creando directorios necesarios...\n")
    setup = SetUp(repetitions, signs=signs)
    Data_Path, actions, video_path = setup.create_directories()
    print(f"Directorios creados en {Data_Path} para las acciones: {actions}")
    logger.info(f"Directorios creados en {Data_Path} para las acciones: {actions}")


def option2(logger, confidence, repetitions, signs, frames, video_paths, mp_path):

    #log
    logger.info("El usuario seleccionó la opción 2 del menú principal")
    #Confidence config
    print("\nAntes de extraer los datos, especifique la confianza del modelo de mediapipe (entre 0 y 1), el valor por defecto es 0.7\n")
    
    user_confidence = input("Ingrese el valor de confianza: ")
    try:
        confidence = float(user_confidence)
        if confidence < 0 or confidence > 1:
            print("Valor fuera de rango, se usará el valor por defecto 0.7\n")
            confidence = 0.7
    except ValueError:
        print("No se ingresó ningún valor, o el valor es inválido, se usará el valor por defecto\n")
        confidence = 0.7


    print(f"Confianza establecida en: {confidence}\n")
    logger.info(f"Confianza del modelo de mediapipe establecida en: {confidence}")

    #Main menu for data extraction
    print("\nExtracción de datos de video: " \
    "Opciones: " \
    "\n1. Extraer datos de un video específico " \
    "\n2. Procesar todos los videos en un directorio" \
    "\n3. Regresar \n")
    user_choice = input("Seleccione una opción: ")
    logger.info(f"El usuario seleccionó {user_choice} en el menú de extracción de datos")

    if user_choice == '1':
        logger.info("El usuario seleccionó la opción 1 en el menú de extracción de datos")
        print("Extrayendo datos de un video específico...")
        video_path = print("No se especifico ningún video, porfavor agregue el video dentro de la variable")
        processor = VideoBatchProcessor(directory=video_path, repetitions=repetitions, signs=signs, frames=frames, confidence=confidence, mp_path=mp_path)
        processor.extract_single_path()
    elif user_choice == '2':
        logger.info("El usuario seleccionó la opción 2 en el menú de extracción de datos")
        print("Extrayendo datos de todos los videos de un directorio padre")
        parent_directory = video_paths
        processor = VideoBatchProcessor(directory=parent_directory, repetitions=repetitions, signs=signs, frames=frames, confidence=confidence, mp_path=mp_path)
        processor.extract_parent_path()


def option3(logger, confidence, repetitions, signs, frames, video_paths, mp_path):
    logger.info("El usuario seleccionó la opción 3 del menú principal")

    #Confidence config
    print("\nAntes de extraer los datos, especifique la confianza del modelo de mediapipe (entre 0 y 1), el valor por defecto es 0.7\n")
    user_confidence = input("Ingrese el valor de confianza: ")
    try:
        confidence = float(user_confidence)
        if confidence < 0 or confidence > 1:
            print("Valor fuera de rango, se usará el valor por defecto 0.7\n")
            confidence = 0.7
    except ValueError:
        print("No se ingresó ningún valor, o el valor es inválido, se usará el valor por defecto\n")
        confidence = 0.7

    print(f"Confianza establecida en: {confidence}\n")
    logger.info(f"Confianza del modelo de mediapipe establecida en: {confidence}")

    #Menu for batch video processing
    print("\nOpciones de Procesamiento de videos: ")
    print("Ojo solo para revisar los videos, no para extraer datos")
    print("1. Extraer datos de un video específico")
    print("2. Procesar todos los videos en un directorio")
    print("3. Regresar")
    user_choice2 = input("Seleccione una opción: ")
    logger.info(f"El usuario seleccionó {user_choice2} en el menú de procesamiento de videos")

    if user_choice2 == '1':
        video_path = print("No se especifico ningún video, porfavor agregue el video dentro de la variable")
        processor = VideoBatchProcessor(directory=video_path, repetitions=repetitions, confidence=confidence, signs=signs, frames=frames, mp_path=mp_path)
        processor.run()
    elif user_choice2 == '2':
        videos_directory = video_paths
        processor = VideoBatchProcessor(videos_directory, repetitions=repetitions, confidence=confidence, signs=signs, frames=frames, mp_path=mp_path)
        processor.train()
    elif user_choice2 == '3':
        print("Regresando al menú principal... \n")
        logger.info("El usuario seleccionó la opción 3 en el menú de procesamiento de videos")
        return
    else:
        print("Opción no válida. Por favor, intente de nuevo. \n")
        return


def option4(logger, repetitions, signs, frames, mp_path):
    logger.info("El usuario seleccionó la opción 4 del menú principal")
    labeller = DataLabelling(repetitions=repetitions, signs=signs, frames=frames, mp_path=mp_path)
    labeller.split_data()


def option5(logger, signs, repetitions, frames, mp_path):
    logger.info("El usuario seleccionó la opción 5 del menú principal")
    training = TrainingLSTM(signs=signs, repetitions=repetitions, frames=frames, mp_path=mp_path)
    training.build_model()


def option6(logger, confidence, signs):
    logger.info("El usuario seleccionó la opción 6 del menú principal")

    #Confidence config
    print("\nAntes de hacer la detección, especifique la confianza del modelo de mediapipe (entre 0 y 1), el valor por defecto es 0.7\n")
    user_confidence = input("Ingrese el valor de confianza: ")
    try:
        confidence = float(user_confidence)
        if confidence < 0 or confidence > 1:
            print("Valor fuera de rango, se usará el valor por defecto 0.7\n")
            confidence = 0.7
    except ValueError:
        print("No se ingresó ningún valor, o el valor es inválido, se usará el valor por defecto\n")
        confidence = 0.7

    print(f"Confianza establecida en: {confidence}\n")
    

    # Real-time detection
    print("Prueba de deteccion: ")
    deteccion = RealtimeDetection(signs=signs, confidence=confidence)
    deteccion.real_time_detection()


def option7(logger):
    logger.info("El usuario seleccionó la opción 7 del menú principal. Saliendo del programa.")
    print("\nSaliendo del programa. ¡Hasta luego!")
    menu = False
    return
#---------------------------------


def main():
    #logger
    logger = Utilities.setup_logging()
    logger.info("Programa iniciado")

    # Configuration
    repetitions = 100
    frames = 30
    signs = ["HOLA", "ADIÓS", "POR-FAVOR", "GRACIAS", "SI", "NO", "BIEN", "MAL", "MAMÁ", "PAPÁ"]
    logger.info(f"Configuración - Repeticiones: {repetitions}, Frames por secuencia: {frames}, Signos: {signs}")
    paths = Utilities.training_paths()
    video_paths = paths[0]
    mp_path = paths[1]

    #Confidence, used for the mediapipe model
    #If the user does not specify a value, the default value will be 0.7
    confidence = 0.7
    logger.info(f"Confianza del modelo de mediapipe establecida en: {confidence}")

    print ("Bienvenido a LETW, el sistema encargado de crear modelos de reconocimiento de lenguaje de señas"\
           "\nPara más información visite: https://github.com/Tonysk8cr/LETW"\
            "\nDesarrollado por @Tonysk8cr \n")

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
        logger.info(f"El usuario seleccionó {user_choice} en el menú principal")
    
        #Dictionary to call the functions
        options = {
            "1": lambda: option1(repetitions, signs, logger),
            "2": lambda: option2(logger, confidence, repetitions, signs, frames, video_paths, mp_path),
            "3": lambda: option3(logger, confidence, repetitions, signs, frames, video_paths, mp_path),
            "4": lambda: option4(logger, repetitions, signs, frames, mp_path),
            "5": lambda: option5(logger, signs, repetitions, frames, mp_path),
            "6": lambda: option6(logger, confidence, signs),
            "7": lambda: option7(logger)
        }

    

        menu_option = options.get(user_choice)
        if menu_option:
            result = menu_option()
            if result is False:
                menu = False

        else:
            print("\nOpción no válida. Por favor, intente de nuevo. \n")
            logger.warning(f"Opción no válida seleccionada: {user_choice}")



if __name__ == '__main__':
    main()
