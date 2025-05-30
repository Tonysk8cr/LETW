# Developed by Anthony Villalobos 08/01/2025
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from matplotlib import pyplot as plt

"""
This class is mainly used to extract image and keypoints also as a way to test the camera and
the enviroment we are working on, it is incharge of doing the image processing, detection, and adding keypoints
to the image.
"""
class ProcesamientoImagen:
    """
    Class used to process the image and extract the keypoints
    """
    def __init__(self):
        #Mediapipe utilities
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(self, image, model):
        """
        Function to make predictions on the image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        """
        Function to draw the landmarks on the image
        """
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.face_landmarks, 
                self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(color=(103,207,245), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
            )
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=3)
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=3)
            )

    def image_processing(self):
        """
        Function to open camera and process the image
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("No se pudo acceder a la cámara.")

          #Instructions text
            print("Instrucciones:")
            print("Coloque su rostro y manos en el cuadro de la cámara.")
            print("Mantenga la manos visibles para una mejor detección.")
            print("Presione 'q' para salir.")

            last_valid_result = None
            last_valid_frame = None

        

            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("No se pudo acceder a la cámara.")
                        break
                    
                    image, results = self.mediapipe_detection(frame, holistic)

                    #Here we save the last valid frame and result
                    if results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                        last_valid_result = results
                        last_valid_frame = frame
                        print("Detectado: ", end="")
                        if results.pose_landmarks: print("Pose ", end="")
                        if results.face_landmarks: print("Cara ", end="")
                        if results.left_hand_landmarks: print("Mano izq ", end="")
                        if results.right_hand_landmarks: print("Mano der ", end="")
                        print("")

                    
                    #Here we add the result of the detection
                    status_text = "Estado: "
                    if results.pose_landmarks:
                        status_text += "Pose detectada"
                    if results.face_landmarks:
                        status_text += ", Cara detectada"
                    if results.left_hand_landmarks:
                        status_text += ", Mano izquierda detectada"
                    if results.right_hand_landmarks:
                        status_text += ", Mano derecha detectada"

                    cv2.putText(image, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    self.draw_styled_landmarks(image, results)
                    cv2.imshow('Captura de video', image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return last_valid_frame, last_valid_result #Return the last valid frame and result
                    
            cap.release()
            cv2.destroyAllWindows()
            return last_valid_frame, last_valid_result #We return the last valid frame and result in case q wasn't pressed
        except Exception as e:
            print(e)
            return None, None

    def last_frame(self, frame):
        """
        Displays the last frame
        """
        if frame is not None:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()
            return True
        return False
    
    #Extract the keypoints
    def extract_keypoints(self, results):
        try:
            #Extract the pose keypoints
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            #Extract the face keypoints
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            #Extract the left hand keypoints
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            #Extract the right hand keypoints
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            #Concatenate all the keypoints
            keypoints = np.concatenate([pose, face, lh, rh])
            return keypoints, True
        except Exception as e:
            print(f"Error al extraer los keypoints: {e}")
            return None, False

def TrainingData():

    def __init__(self):
        #Mediapipe utilities
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
    
    #I know I said in S_Task this was going to stay there but i changed my mind
    #Even though this is pretty much the same as the one in S_Task, this one is going to be used to create the dataset
    def directory_creation(self):
        Data_Path = os.path.join("MP_Data")
        #Here´s the array that contains all the words in LESCO, we need this as we
        #are creating a folder for each word
        #It is called actions beacuse in each folder we will store the images of the actions
        actions = np.array([
            "A-PARTIR-DE", "A-VECES", "A-VECES-2", "ABOGADO", "ABRAZAR", "ABRIL", "ABRIR", "ABRIR-2", "ABUELO", "ABURRIR",
            "ACCESO/ACCEDER", "ACCIDENTE", "ACEPTAR", "ACOMPANAR", "ACONGOJARSE", "ACONSEJAR", "ACOSTUMBRAR", "ACTIVIDAD", "ACUERDO", "ADELANTAR",
            "ADIOS/SALUDAR", "ADJUNTAR-UN-ARCHIVO", "ADULTO", "AFILIAR", "AFRICA", "AGOSTO", "AGREGAR", "AGRUPARSE", "AGUA", "AGUA-DULCE",
            "AHORA", "AHORITA", "AHORRAR", "AISLADO-EN-CASA", "AL-FRENTE", "ALAJUELA", "ALAJUELITA", "ALBANIA", "ALEMANIA/ALEMAN", "ALGUNO/ALGO",
            "ALIVIO/ALIVIARSE", "ALMA", "ALQUILER", "AMAR/AMOR", "AMARILLO", "AMBOS", "AMERICA", "AMETRALLAR", "AMIGO", "ANALIZAR/EXAMINAR",
            "ANARANJADO", "ANASCOR", "ANGEL", "ANIMAL", "ANO", "ANTES", "APARECER", "APARIENCIA", "APARTAR", "APARTE",
            "APELLIDO", "APESTAR", "APLAUDIR", "APLICAR", "APOYAR", "APRENDER", "APROBAR", "APROVECHAR", "ARABIA-SAUDI", "ARABIA/ARABE",
            "ARBITRO", "ARBOL", "ARREGLAR", "ARREPENTIRSE", "ASAMBLEA", "ASAMBLEA-LEGISLATIVA", "ASERRI", "ASIA", "ASOCIACION", "ASOCIADO",
            "ASUSTAR", "ASUSTARSE", "ATENDER", "ATERRIZAR/VENIR-EN-AVION", "ATRAS", "ATRASAR", "AUDIFONO", "AVISAR", "AYER", "AYUDAR",
            "AZUL", "BACHILLERATO", "BAILAR", "BALLENA", "BALLENA-2", "BANANO", "BANARSE", "BANCO", "BANO", "BARATO",
            "BARCO/IR-EN-BARCO", "BARVA", "BASURA", "BEBE", "BENEFICIARSE", "BIBLIA", "BIBLIOTECA", "BICICLETA", "BIENVENIDO", "BILINGUISMO",
            "BIOLOGIA", "BLANCO", "BLUSA", "BODA", "BOLIVIA", "BONITO", "BOSQUE", "BOSTON", "BRASIL", "BRAVO",
            "BRIBRI", "BRILLAR", "BRINCAR", "BROMA", "BRUTO", "BUENO", "BUS", "BUSCAR", "CABALLO", "CABEZA",
            "CADA-UNO", "CAERSE", "CAERSE-2", "CAFE (BEBIDA)", "CAFE (COLOR)", "CAJA", "CALIENTE", "CALOR", "CALZONCILLOS", "CAMA",
            "CAMARON", "CAMBIAR", "CAMINAR", "CAMION/MANEJAR-CAMION", "CAMISA", "CANADA", "CANCER", "CANSAR", "CARA", "CARACTER",
            "CARIBE", "CARNE", "CARRERA (ACADEMICA)", "CARRO/MANEJAR-CARRO", "CARTA", "CARTAGO", "CASA", "CASTIGAR", "CATOLICO", "CATORCE",
            "CEBOLLA", "CELESTE", "CELOS/CELOSO", "CENAREC", "CENTRO", "CENTROAMERICA", "CERCA", "CERDO", "CERO", "CERRAR",
            "CERTIFICADO", "CERVEZA", "CHILE (FRUTO)", "CHILE (PAIS)", "CHINA", "CHINO", "CHOFER", "CIEGO", "CIEN", "CIEN-2",
            "CIENCIAS", "CINCO", "CINCUENTA", "CIUDAD-QUESADA", "CLARO", "CLASE", "CLIMA", "CLINICA", "COCA-COLA", "COCINA",
            "COCINAR", "COCODRILO", "COLABORAR", "COLOMBIA", "COLON", "COLOR", "COMA", "COMER", "COMO", "COMO-2",
            "COMODO", "COMPANERO", "COMPARAR", "COMPARTIR", "COMPLICAR", "COMPRAR", "COMPUTADORA", "COMUNICARSE/COMUNICACION", "COMUNIDAD", "CON-CIERTA-FRECUENCIA",
            "CON-EL-TIEMPO", "CON-GUSTO", "CONOCER", "CONSOLAR", "CONSTRUIR", "CONTACTAR", "CONTAR", "CONTINUAR/SEGUIR", "CONTRA", "CONTRATAR",
            "CONTROLAR", "COPIAR", "CORONADO", "CORRECTO", "CORREO-ELECTRONICO", "CORRER", "CORRER-2", "CORTAR-RELACION", "CORTO/BREVE", "COSTA-RICA",
            "CRECER", "CREER", "CRUZ-ROJA", "CUAL", "CUAL-2", "CUALQUIERA/NO-IMPORTAR", "CUANDO", "CUANTO", "CUARENTA", "CUARTO",
            "CUATRO", "CUATRO-ADJETIVO", "CUATROCIENTOS", "CUBA", "CUCARACHA", "CUCHARA", "CUCHILLO", "CUERPO", "CUIDAR-2", "CULANTRO",
            "CULTURA", "CUMPLEANOS", "CUMPLIR", "CUNADO", "CURIOSIDAD/CURIOSO", "CURRIDABAT", "CURSO", "DAR", "DARSE-CUENTA", "DE-LEJOS",
            "DE-PIE", "DEBAJO", "DEBER", "DEBIL", "DECIDIR", "DECIMO", "DECIR", "DECIR-2", "DEFECAR", "DEJAR",
            "DEJAR-2", "DELETREAR", "DELFIN", "DEMANDAR/DENUNCIAR", "DENTRO", "DEPENDER", "DEPORTE", "DERECHO", "DESAMPARADOS", "DESAPARECER",
            "DESAPARECER-2", "DESARROLLAR", "DESARROLLAR-2", "DESCANSAR", "DESDE", "DESDE-ANTES-HASTA-AHORA", "DESDE-ANTES-HASTA-CIERTO-MOMENTO", "DESDE-CIERTO-MOMENTO-HACIA-ADELANTE", "DESDE-CIERTO-MOMENTO-HACIA-ATRAS", "DESMORONAR",
            "DESORDEN", "DESPACIO", "DESPEDIR", "DESPEGAR-AVION", "DESPERTARSE", "DESPLAZARSE", "DESPUES", "DESTRUIR", "DIA", "DIABLO",
            "DIBUJAR", "DICCIONARIO", "DICIEMBRE", "DIECINUEVE", "DIECIOCHO", "DIECISEIS", "DIECISIETE", "DIEZ", "DIFERENTE/SER-DIFERENTE", "DIFICIL",
            "DIMINUTO", "DIOS", "DIPUTADO", "DIRECCION", "DIRECTOR", "DISCAPACIDAD", "DISCO", "DISCRIMINAR", "DISCUTIR", "DISENAR",
            "DISFRUTAR/DIVERTIRSE", "DISIMULAR", "DISMINUIR", "DIVISION/DIVIDIR", "DIVORCIO", "DOCE", "DOCTOR", "DOLAR", "DOLOR", "DOMINGO",
            "DONDE", "DORMIR", "DOS", "DOS-ADJETIVO", "DOSCIENTOS", "DUDAR", "DULCE (CARACTER)", "DULCE (SABOR)", "DURANTE-ESTE-TIEMPO", "DURAR",
            "DURO/ESTRICTO", "ECONOMIA", "ECUADOR", "EDITAR", "EDUCACION", "EJEMPLO", "EL-SALVADOR", "ELECTRICIDAD", "ELEFANTE", "EMOCIONARSE",
            "EMPATAR", "EMPEZAR/COMENZAR", "EMPLEADO", "EMPRESA/FABRICA", "EMPUJAR", "EN-LA-MANANA", "EN-LA-TARDE", "EN-SEGUNDO-LUGAR", "EN-SEGUNDO-LUGAR-2", "EN-TERCER-LUGAR-2",
            "ENAMORARSE", "ENAMORARSE-2", "ENCAJAR", "ENCONTRARSE-CON", "ENERO", "ENFATIZAR", "ENFERMO", "ENFRENTARSE", "ENGORDAR", "ENSALADA",
            "ENSENAR", "ENTENDER", "ENTONCES", "ENTRAR", "ENTRE-COMILLAS", "ENTRE-OTROS", "ENTREGAR-RECIBIR", "ENTRENADOR", "ENTREVISTAR", "ENVIAR",
            "EQUILIBRAR", "EQUIPO", "ERROR/EQUIVOCARSE", "ESCAZU", "ESCLAVO", "ESCOBILLAS", "ESCOGER", "ESCONDER/ESCONDIDO", "ESCONDERSE", "ESCRIBIR",
            "ESCRIBIR-EN-COMPUTADORA", "ESCUCHAR", "ESCUELA", "ESFORZAR", "ESO", "ESPANA", "ESPANOL", "ESPECIAL", "ESPERAR", "ESPERE!",
            "ESPIRITU", "ESPOSO", "ESPOSO-2", "ESQUINA", "ESTABLECER", "ESTADIO", "ESTADOS-UNIDOS", "ESTAR-BALANCEADO", "ESTAR-BIEN", "ESTAR-BIEN-2",
            "ESTAR-CONFUNDIDO", "ESTAR-ESTRECHO-ECONOMICAMENTE", "ESTAR-PRESENTE-DANDO-LA-CARA", "ESTAR-SATISFECHO", "ESTAR-TENTADO", "ESTATUTO", "ESTE", "ESTOMAGO", "ESTUDIAR", "ESTUDIOS-SOCIALES",
            "EUROPA", "EXAMEN", "EXCELENTE", "EXCELENTE-2", "EXPANDIRSE", "EXPEDIENTE", "EXPERIMENTAR", "EXPLICAR", "EXPLOTAR", "EXPONER",
            "EXPRESAR", "EXPULSAR", "EXTENSION-ESPACIO", "FACEBOOK", "FACIL", "FACTURA", "FALTAR", "FALTAR-AIRE", "FAMILIA", "FAMOSO",
            "FEBRERO", "FELICITAR", "FELIZ/CONTENTO/ALEGRE", "FEO", "FIEL", "FIESTA", "FILA", "FINCA", "FIRMAR", "FISCAL",
            "FISICA", "FLEXIBLE", "FLOR", "FLUIDO", "FORMA", "FORMARSE/SURGIR", "FORMATO", "FOTO", "FRACCION", "FRANCIA",
            "FRASE", "FRENTE(CUERPO)", "FRESCO", "FRIO", "FRUSTRAR", "FRUTA", "FRUTA-2", "FUERTE", "FUNCION", "FUNDAR",
            "FUTURO", "GALLETA", "GALLINA", "GANAR", "GASOLINA", "GASTAR", "GATO", "GESTO", "GIMNASIO", "GOBIERNO",
            "GOL", "GOLFITO", "GOLPEAR", "GOLPEARSE", "GORDO", "GORILA", "GRABAR-SONIDO", "GRACIAS", "GRACIAS-A-DIOS", "GRADO",
            "GRADUARSE", "GRAMATICA", "GRANDE", "GRANDE-2", "GRATIS", "GRECIA", "GRIPE", "GRIS", "GRITAR", "GRUA",
            "GRUPO", "GUADALUPE", "GUANACASTE", "GUANTES", "GUAPILES", "GUARDAR", "GUATEMALA", "GUATUSO", "GUERRA", "GUIA/GUIAR",
            "GUSTAR", "HABER/HAY", "HABLAR", "HACER-A-UN-LADO", "HAITI", "HARTO", "HASTA", "HATILLO", "HAY-DIFERENTES", "HELADO",
            "HELICOPTERO", "HEREDIA", "HERMANO", "HIGADO", "HIJA", "HIJO", "HIJO-2", "HIJUEPUTA", "HISTORIA", "HOJA (PLANTA)",
            "HOLA", "HOMBRE", "HONDURAS", "HONESTO", "HORARIO", "HOSPITAL", "HOTEL", "HOY", "HUELGA", "HUEVO",
            "HUMILDE", "HUNDIRSE/DECRECER", "IDENTIDAD", "IDIOMA", "IGNORELO!", "IMAGEN", "IMAGINAR", "IMPLANTE-COCLEAR", "IMPORTANTE", "IMPOTENTE",
            "IMPRIMIR", "INCLUIR", "INDEPENDIZAR", "INDIA", "INDIO", "INFORMAR", "INFORMAR-2", "INGENIERO", "INGLATERRA", "INGLES",
            "INOCENTE", "INSISTIR", "INSTITUTO", "INTEGRAR/INTEGRACION", "INTELIGENTE", "INTERACTUAR", "INTERCAMBIAR", "INTERESANTE", "INTERNACIONAL", "INTERPRETE",
            "INTERRUMPIR", "INVENTAR", "INVESTIGAR", "INVIERNO", "INVITAR", "INYECTAR", "IR", "IR-2", "IR-3", "IRSE",
            "IRSE-2", "IRSE-3", "ISLA", "ITALIA", "JACO", "JAMAICA", "JAMAS", "JAPON", "JESUS", "JIRAFA",
            "JOVEN", "JUEVES", "JUGADOR", "JUGAR", "JULIO", "JUNIO", "JUNTOS/CON", "KINDER", "KINDER-2", "LA-FORTUNA",
            "LA-SABANA", "LAPIZ", "LASTIMA", "LATINOAMERICA", "LECCION", "LECHE", "LECTURA", "LEER", "LENGUA", "LEON",
            "LESCO", "LETRA", "LETRA-A", "LETRA-B", "LETRA-C", "LETRA-CH", "LETRA-D", "LETRA-E", "LETRA-F", "LETRA-G",
            "LETRA-H", "LETRA-I", "LETRA-J", "LETRA-K", "LETRA-L", "LETRA-LL", "LETRA-M", "LETRA-N", "LETRA-N", "LETRA-O",
            "LETRA-P", "LETRA-Q", "LETRA-R", "LETRA-RR", "LETRA-S", "LETRA-T", "LETRA-U", "LETRA-V", "LETRA-W", "LETRA-X",
            "LETRA-Y", "LETRA-Z", "LEY", "LIBERIA", "LIBRE", "LIBRO", "LIMITADO", "LIMON (FRUTA)", "LIMON (LUGAR)", "LINEA",
            "LISTA", "LISTO", "LLAMAR-POR-TELEFONO", "LLAVE", "LLEVAR-TRAER", "LLORAR", "LLOVER", "LOCO", "LOGICO", "LUCHAR",
            "LUGAR", "LUNA", "LUNES", "LUNES-A-VIERNES", "LUZ", "MACARRONES", "MADERA", "MADRUGADA", "MADURAR/MADUREZ", "MAL",
            "MAL-2/ESTAR-MAL", "MAMA", "MANANA", "MANO", "MANO-2", "MAQUINA", "MAR", "MARCHA", "MAREO", "MARTES",
            "MARTILLO", "MARZO", "MAS", "MAS-ADELANTE", "MAS-O-MENOS", "MAS-SUMA", "MATAR", "MATEMATICAS", "MAYO", "MAYORIA",
            "MEDIAS", "MEDICINA", "MEJOR", "MEMORIA", "MENSAJE-DE-TEXTO", "MENTIR", "MERCADO-CENTRAL", "MES", "MESA", "MEXICO",
            "MICROONDAS", "MIEDO", "MIERCOLES", "MIL", "MILLON", "MINIMO", "MINISTERIO/MINISTRO", "MINUTO", "MITAD/MEDIA", "MODELO",
            "MOJADO", "MONJA", "MONJE", "MONTANA", "MORADO", "MORAVIA", "MORENO", "MORIR", "MOSTRAR", "MOTIVAR",
            "MUCHO", "MUDARSE", "MUDO", "MUJER", "MUJER-2", "MULTIPLICAR", "MUNDO", "MUNDO-2", "MUSULMAN", "MUY-POCO",
            "NACER", "NADA", "NADA-MAS", "NADAR", "NARANJA", "NARIZ", "NECESITAR", "NEGRO", "NEGRO (ETNIA)", "NERVIOSO",
            "NICARAGUA", "NICOYA", "NINO", "NINO-2", "NO", "NO-2", "NO-3", "NO-4", "NO-ENTENDER-NADA", "NO-HABER-CAMPO",
            "NO-HAY/NO-HABER", "NO-PODER", "NO-PODER-DORMIR", "NO-SABER-QUE-HACER", "NO-SE/NO-SABER", "NOCHE", "NOMBRE", "NORMAL", "NORTE", "NOSOTROS-INDEF-1",
            "NOSOTROS-INDEF-2", "NOTA", "NOTICIA", "NOVECIENTOS", "NOVENO", "NOVENTA", "NOVIEMBRE", "NOVIO", "NUEVE", "NUEVO",
            "NUMERO", "NUNCA", "O", "OBJETIVO", "OBLIGAR", "OCHENTA", "OCHO", "OCHOCIENTOS", "OCTAVO", "OCTUBRE",
            "OCTUBRE-2", "OCUPADO", "ODIAR", "OESTE", "OFICIAL", "OFICINA", "OJALA", "OK", "ONCE", "OPINAR",
            "ORALIZAR", "ORINAR", "OSCURO", "OTRA-VEZ", "OTRO", "OVNI", "OYENTE", "PACIENCIA/AGUANTAR", "PADRES", "PADRES-2",
            "PAGAR", "PAGINA", "PAIS", "PAJARO", "PALABRA", "PALMARES", "PALMARES-2", "PAN", "PANAMA", "PAPA",
            "PAPAS", "PARA", "PARA-QUE", "PARAGUAS", "PARAGUAY", "PARAISO", "PARALISIS-CEREBRAL", "PARAR-VEHICULO", "PARECER", "PARQUE",
            "PASADO", "PASEAR", "PATINES", "PATO-2", "PAYASO", "PAZ", "PECADO", "PEDIR", "PELIGROSO", "PENSAR",
            "PENSAR-2/ACORDARSE-DE", "PENSAR-3", "PEOR", "PEQUENO(ENTIDAD HORIZONTAL)", "PEQUENO(ENTIDAD VERTICAL)", "PERCIBIR", "PERDER", "PERDER/REPROBAR", "PEREZ-ZELEDON", "PERFECTO",
            "PERIODICO", "PERMISO", "PERRO", "PERSONA", "PERU", "PEZ", "PIES", "PIRAMIDE", "PIZZA", "PLAYA",
            "PLAZA-DE-LA-CULTURA-1", "PLAZA-DE-LA-CULTURA-2", "PLAZA-VIQUEZ", "POBRE", "POCO", "POCO-A-POCO", "PODER", "POLLITO", "POLLO", "POLLO-2",
            "PONER", "POR-CIENTO/PORCENTAJE", "POR-ESO", "POR-FAVOR", "POR-SI-SOLO", "PORQUE", "POSTERGAR", "PRACTICAR", "PREFERIR", "PREGUNTAR",
            "PREMIO", "PREOCUPARSE", "PRESTAR", "PRIMERA-VEZ", "PRIMERO", "PRIMO", "PRIVADO", "PRO.POSESIVO", "PRO1", "PRO1.POSESIVO",
            "PRO2", "PROBAR", "PROBLEMA", "PRODUAL-2", "PRODUAL.1", "PROFESOR", "PROFUNDO", "PROMISMO", "PROPIO", "PROPLURAL",
            "PROPLURALDISTRIBUTIVO", "PROPONER", "PROTESTAR/QUEJARSE", "PROTRIAL.1", "PROVINCIA", "PSICOLOGIA", "PUENTE", "PUERTO-RICO", "PULPERIA", "PUNTARENAS",
            "QUE", "QUE-HACER?", "QUE-PASAR", "QUEDARSE", "QUEQUE", "QUERER", "QUIEN", "QUIMICA", "QUINCE", "QUINIENTOS",
            "QUINTO", "QUIZ", "RAPIDO", "RARO", "RATON", "RAZON", "RECONOCER", "RECORDAR", "REDUCIRSE", "REGANAR",
            "REGANAR-DURO", "REGLA", "REINA", "REIRSE", "RELIGION", "RESOLVER", "RESPETAR", "RESPONDER", "RESPONSABLE", "RESTAR",
            "RESTAURANTE", "REY", "RIO", "ROJO", "ROMANO", "ROSADO", "SABADO", "SABER", "SACERDOTE", "SALIR",
            "SAN-CARLOS", "SAN-JOSE", "SAN-PEDRO", "SAN-RAMON", "SANAR/SALUD", "SANTA-LUCIA", "SANTO-DOMINGO", "SARCHI", "SECO", "SEGUIR-ADELANTE",
            "SEGUNDO", "SEGURO", "SEIS", "SEISCIENTOS", "SEMANA", "SEMBRAR", "SEMINARIO", "SENA", "SENA-2", "SENA-3",
            "SENA-4", "SENTARSE", "SENTIR", "SER-ALTO/ESTAR-GRANDE", "SER-IMPOSIBLE", "SER-PRO1", "SER-PRO2", "SER-TIPICO-DE", "SERVIR", "SESENTA",
            "SETECIENTOS", "SETENTA", "SETIEMBRE", "SETIMO", "SEXO", "SEXTO", "SI", "SIEMPRE", "SIETE", "SIGNIFICAR",
            "SILLA", "SIMBOLO", "SIMPLE", "SINDROME-DE-DOWN", "SITUACION", "SOBRE", "SOBRINO", "SODA", "SOL", "SOLO",
            "SOPA", "SORDO", "SORDO^MUDO", "SORPRENDIDO", "SOSTEN", "SUAVE", "SUBTITULOS", "SUCEDER/PASAR-ALGO", "SUCIO", "SUENO",
            "SUERTE", "SUFRIR", "SUMAR", "SUR", "TAL-VEZ", "TAMAL", "TAMBIEN", "TAMBIEN-2", "TAREA", "TAXI",
            "TE", "TE^FRIO", "TEMA", "TEMBLAR", "TENEDOR", "TENER", "TENER-CULPA", "TENER-EXPERIENCIA", "TENER-RAZON", "TENER-SUERTE",
            "TERCERO", "TERMINAR/ULTIMO", "TERRIBLE", "TIBAS", "TIEMPO-2", "TIEMPO/HORA", "TIERRA(MATERIAL)", "TIERRA(PLANETA)", "TIO", "TIO-2",
            "TOCAR", "TODA-LA-MANANA", "TODAVIA-NO", "TODO", "TODO-EL-DIA", "TODOS-LOS-DOMINGOS", "TODOS-LOS-JUEVES", "TODOS-LOS-LUNES", "TODOS-LOS-MARTES", "TODOS-LOS-MIERCOLES",
            "TODOS-LOS-SABADOS", "TODOS-LOS-VIERNES", "TOMAR", "TOMATE", "TORTA", "TORTUGA", "TRABAJAR", "TRATAR", "TRATAR-DE-COMUNICARSE", "TRATAR-DE-CONVERSAR",
            "TRAUMA", "TRECE", "TREINTA", "TRES", "TRES-RIOS", "TRESCIENTOS", "TURISMO", "TURQUIA", "TURRIALBA", "TURRUCARES",
            "UNICO", "UNIVERSIDAD", "UNO", "UNO-MITAD", "URUGUAY", "USAR", "VACA/TORO", "VACACIONES", "VACIO", "VAGABUNDO",
            "VAPOR", "VEGETAL", "VEINTE", "VENCER", "VENDER", "VENEZUELA", "VENIR", "VENTANA", "VER", "VER-2",
            "VER-3", "VERANO", "VERDAD", "VERDE", "VEZ", "VIDA", "VIEJO", "VIENTO", "VIERNES", "VIOLAR-LA-LEY",
            "VOCABULARIO", "VOLAR-EN-AVION", "VOLCAN", "VOZ", "WINDOWS", "WWW", "YA", "YA-2", "YUCA", "ZACATE",
            "ZAPATO", "ZAPOTE", "ZARCERO", "ZARCERO-2"
        ])
        #30 videos of data
        number_sequences = 30
        #30 frames of length
        Sequence_length = 30

        return Data_Path, actions, number_sequences, Sequence_length


    def mediapipe_detection(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("No se pudo acceder a la cámara.")
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            print("No se pudo acceder a la cámara.")
                            break
                        #Make detections
                        image, results = self.mediapipe_detection(frame, holistic)

                        #Draw landmarks
                        self.draw_styled_landmarks(image, results)
                        
                        #Show to screen
                        cv2.imshow('Captura de video', image)

                        #Exit
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(e)
            return None, False
          
def main():
    procesador = ProcesamientoImagen()
    frame, results = procesador.image_processing() #Here we receive the frame and results
    if results is not None:
        keypoints, success = procesador.extract_keypoints(results)
        if success:
            print(f"Keypoints extraidos: {keypoints}")
            print(f"Número de keypoints: {len(keypoints)}")	
        else:
            print("No se pudo extraer los keypoints")
    else:
        print("No se pudo obtener los resultados")

if __name__ == '__main__':
    main()