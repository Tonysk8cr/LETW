class Strings:
    class MainMenu:
        WELCOME_MESSAGE = (
            "Bienvenido a LETW, el sistema encargado de crear modelos de reconocimiento de lenguaje de señas"
            "\nPara más información visite: https://github.com/Tonysk8cr/LETW"
            "\nDesarrollado por @Tonysk8cr \n"
        )
        HEADER = "Hola, seleccione una opción:"
        OPTION_CREATE_DIRECTORIES = "Crear directorios necesarios"
        OPTION_EXTRACT_DATA = "Procesar y extraer datos de video"
        OPTION_REVIEW_BATCH = "Procesar videos en lote"
        OPTION_LABEL_DATA = "Label Data"
        OPTION_TRAIN_MODEL = "Train LSTM"
        OPTION_REALTIME_DETECTION = "Detección en tiempo real"
        OPTION_EXIT = "Salir"
        INPUT_OPTION = "Ingrese su opción ({}-{}): "
        INVALID_OPTION = "\nOpción no válida. Por favor, intente de nuevo. \n"

    class CreateDirs:
        CREATING = "Creando directorios necesarios...\n"
        CREATED = "Directorios creados en {} para las acciones: {}"

    class Confidence:
        PROMPT = (
            "\nAntes de extraer los datos, especifique la confianza del modelo de mediapipe (entre 0 y 1), el valor por defecto es {}"
        )
        PROMPT_DETECTION = (
             "\nAntes de hacer la detección, especifique la confianza del modelo de mediapipe (entre 0 y 1), el valor por defecto es {}\n"
        )
        INPUT = "Ingrese el valor de confianza: "
        OUT_OF_RANGE = "Valor fuera de rango, se usará el valor por defecto {}\n"
        INVALID_INPUT = "No se ingresó ningún valor, o el valor es inválido, se usará el valor por defecto\n"
        SET_MSG = "Confianza establecida en: {}\n"

    class ExtractData:
        MENU = (
            "\nExtracción de datos de video: "
            "Opciones: "
            "\n1. Extraer datos de un video específico "
            "\n2. Procesar todos los videos en un directorio"
            "\n3. Regresar \n"
        )
        INPUT_OPTION = "Seleccione una opción: "
        EXTRACTING_SPECIFIC = "Extrayendo datos de un video específico..."
        NO_VIDEO_SPECIFIED = "No se especifico ningún video, porfavor agregue el video dentro de la variable"
        EXTRACTING_ALL = "Extrayendo datos de todos los videos de un directorio padre"

    class ReviewBatch:
        MENU = (
            "\nOpciones de Procesamiento de videos: "
            "\nOjo solo para revisar los videos, no para extraer datos"
            "\n1. Extraer datos de un video específico"
            "\n2. Procesar todos los videos en un directorio"
            "\n3. Regresar"
        )
        INPUT_OPTION = "Seleccione una opción: "
        RETURNING_MAIN = "Regresando al menú principal... \n"

    class RealtimeDetection:
        TEST_MSG = "Prueba de deteccion: "

    class Exit:
        GOODBYE = "\nSaliendo del programa. ¡Hasta luego!"
