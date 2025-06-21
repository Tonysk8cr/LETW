# Developed by Anthony Villalobos 08/01/2025
# Adapted to use a VIDEO instead of the camera
#Updated by Anthony Villalobos 02/06/2025


from VideoBatchProcessor import VideoBatchProcessor
from DataExtraction import DataExtractor
from DataLabelling import DataLabelling
from TrainingLSTM import TrainingLSTM

def main():
    print("Hola, seleccione una opción:")
    print("1. Extraer datos de video")
    print("2. Procesar videos en lote")
    print("3. Test")
    print("4. Train LSTM")
    print("5. Salir")


        #Notaaaaaa ir a DataExtraction y buscar current_transofrm, pasarlo una vez bajo none, y despues por Utilities.random_augmentation
    user_choice = input("Ingrese su opción (1/2/3/4)")
    if user_choice == '1':
        video_path = r"C:\Users\tonyi\LETW\Test\Test_Videos"
        processor = VideoBatchProcessor(directory=video_path)
        processor.extract()
    elif user_choice == '2':
        videos_directory = r"C:\Users\tonyi\LETW\Test\Test_Videos"
        processor = VideoBatchProcessor(videos_directory)
        processor.run()
    elif user_choice == '3':
        labeller = DataLabelling()
        labeller.split_data()
    elif user_choice == '4':
        training = TrainingLSTM()
        training.build_model()
    elif user_choice == '5':
        print("Saliendo del programa. ¡Hasta luego!")
        return
    else:
        print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == '__main__':
    main()