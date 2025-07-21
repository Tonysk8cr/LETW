"""
This file is used to create the tasks that are going to be executed only once
Mostly secundary tasks that are not going to be executed in the main loop
For example, the directory creation etc.
Done like this to automate a bit the process
"""


#Creation of fodlers
import os
import numpy as np
import time
import pyautogui
import sys

def directory():
    Data_Path = os.path.join("MP_Data")
    #Here´s the array that contains all the words in LESCO, we need this as we
    #are creating a folder for each word
    #It is called actions beacuse in each folder we will store the images of the actions
    actions = np.array(["ANASCOR","A-PARTIR-DE","CAERSE","CALZONCILLOS", "CIUDAD-QUESADA", "NOTA"])
    #30 videos of data
    number_sequences = 50
    #30 frames of length
    Sequence_length = 30

    #This loop is the one in charge of making each one of the directories
    for action in actions:
        #For each one of the main folder we will create 30 more folders from "0" to "29"
        for sequence in range(number_sequences):
            try:
                os.makedirs(os.path.join(Data_Path, action, str(sequence)))
            except:
                pass
    print("Finalizado")
    return Data_Path, actions, number_sequences, Sequence_length

def clicker():
    # Pedir al usuario el tiempo entre clics
    delay = float(input("Ingresa el tiempo (en segundos) entre cada clic: "))
    
    print("Tienes 5 segundos para mover el ratón a la posición deseada...")
    time.sleep(5)

    print("Autoclicker iniciado. Presiona Ctrl + C para detener.")
    try:
        while True:
            pyautogui.click()
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nAutoclicker detenido.")


def namefiles():

    def listar_archivos_en_texto(directorio, archivo_salida="lista_archivos.txt"):
        """
        Crea un archivo de texto con la lista de archivos y carpetas
        dentro del directorio especificado.
        """
        elementos = os.listdir(directorio)
        # Ordenar alfabéticamente (opcional)
        elementos.sort()

        with open(archivo_salida, "w", encoding="utf-8") as f:
            for item in elementos:
                f.write(item + "\n")

        print(f"Se generó el archivo {archivo_salida} con la lista de archivos.")

    # Llamada directa a la función
    ruta_directorio = r"C:\Users\tonyi\OneDrive\Documentos\LETW\T_Videos"
    listar_archivos_en_texto(ruta_directorio)


def detectar_archivos_faltantes(directorio, nombres_esperados, extension=".mp4"):
    """
    Verifica la existencia del directorio, lista los archivos con la extensión dada,
    normaliza los nombres y devuelve una lista con los nombres faltantes.
    """
    if not os.path.isdir(directorio):
        print(f"ERROR: El directorio '{directorio}' no existe.")
        return None

    try:
        archivos_existentes = [f for f in os.listdir(directorio) if f.lower().endswith(extension.lower())]
    except Exception as e:
        print(f"ERROR al listar archivos en '{directorio}': {e}")
        return None

    # Normalizar nombres: se elimina la extensión y se convierten a mayúsculas
    archivos_existentes_norm = [os.path.splitext(f)[0].upper() for f in archivos_existentes]
    nombres_esperados_norm = [nombre.upper() for nombre in nombres_esperados]

    # Encontrar los nombres que faltan
    faltantes = [nombre for nombre in nombres_esperados if nombre.upper() not in archivos_existentes_norm]

    return faltantes

def generar_reporte(faltantes, total_esperados, directorio):
    """
    Muestra en consola un reporte con la cantidad total de archivos esperados, 
    la cantidad de archivos faltantes, el porcentaje completado y la lista de archivos faltantes.
    """
    print("\n===== REPORTE DE ARCHIVOS FALTANTES =====")
    print(f"Directorio analizado: {directorio}")
    print(f"Total de archivos esperados: {total_esperados}")
    print(f"Total de archivos faltantes: {len(faltantes)}")
    if total_esperados > 0:
        porcentaje = (total_esperados - len(faltantes)) / total_esperados * 100
        print(f"Porcentaje completado: {porcentaje:.2f}%")
    else:
        print("No se especificaron archivos esperados.")
    
    if faltantes:
        print("\nLista de archivos faltantes:")
        for i, nombre in enumerate(faltantes, 1):
            print(f"{i}. {nombre}")
    else:
        print("\n¡Felicidades! No falta ningún archivo.")
    print("=========================================")

def guardar_lista_faltantes(faltantes, nombre_archivo="archivos_faltantes.txt"):
    """
    Guarda la lista de archivos faltantes en un archivo de texto.
    """
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            for nombre in faltantes:
                f.write(f"{nombre}\n")
        print(f"\nLista de archivos faltantes guardada en '{nombre_archivo}'")
        return True
    except Exception as e:
        print(f"\nERROR al guardar la lista de faltantes: {e}")
        return False
    #This should be copied on main, in case we want to use this fumction
        # Ejemplo de nombres esperados, modifícalos según tus necesidades
    nombres_esperados = [
    "A-PARTIR-DE", "A-VECES", "A-VECES-2", "ABOGADO", "ABRAZAR", "ABRIL", "ABRIR", "ABRIR-2", "ABUELO", "ABURRIR",
    "ACCESO/ACCEDER", "ACCIDENTE", "ACEPTAR", "ACOMPAÑAR", "ACONGOJARSE", "ACONSEJAR", "ACOSTUMBRAR", "ACTIVIDAD", "ACUERDO", "ADELANTAR",
    "ADIÓS/SALUDAR", "ADJUNTAR-UN-ARCHIVO", "ADULTO", "AFILIAR", "AFRICA", "AGOSTO", "AGREGAR", "AGRUPARSE", "AGUA", "AGUA-DULCE",
    "AHORA", "AHORITA", "AHORRAR", "AISLADO-EN-CASA", "AL-FRENTE", "ALAJUELA", "ALAJUELITA", "ALBANIA", "ALEMANIA/ALEMÁN", "ALGUNO/ALGO",
    "ALIVIO/ALIVIARSE", "ALMA", "ALQUILER", "AMAR/AMOR", "AMARILLO", "AMBOS", "AMÉRICA", "AMETRALLAR", "AMIGO", "ANALIZAR/EXAMINAR",
    "ANARANJADO", "ANASCOR", "ANGEL", "ANIMAL", "AÑO", "ANTES", "APARECER", "APARIENCIA", "APARTAR", "APARTE",
    "APELLIDO", "APESTAR", "APLAUDIR", "APLICAR", "APOYAR", "APRENDER", "APROBAR", "APROVECHAR", "ARABIA-SAUDÍ", "ARABIA/ÁRABE",
    "ARBITRO", "ARBOL", "ARREGLAR", "ARREPENTIRSE", "ASAMBLEA", "ASAMBLEA-LEGISLATIVA", "ASERRÍ", "ASIA", "ASOCIACIÓN", "ASOCIADO",
    "ASUSTAR", "ASUSTARSE", "ATENDER", "ATERRIZAR/VENIR-EN-AVIÓN", "ATRÁS", "ATRASAR", "AUDÍFONO", "AVISAR", "AYER", "AYUDAR",
    "AZUL", "BACHILLERATO", "BAILAR", "BALLENA", "BALLENA-2", "BANANO", "BAÑARSE", "BANCO", "BAÑO", "BARATO",
    "BARCO/IR-EN-BARCO", "BARVA", "BASURA", "BEBÉ", "BENEFICIARSE", "BIBLIA", "BIBLIOTECA", "BICICLETA", "BIENVENIDO", "BILINGÜISMO",
    "BIOLOGÍA", "BLANCO", "BLUSA", "BODA", "BOLIVIA", "BONITO", "BOSQUE", "BOSTON", "BRASIL", "BRAVO",
    "BRIBRI", "BRILLAR", "BRINCAR", "BROMA", "BRUTO", "BUENO", "BUS", "BUSCAR", "CABALLO", "CABEZA",
    "CADA-UNO", "CAERSE", "CAERSE-2", "CAFÉ (BEBIDA)", "CAFÉ (COLOR)", "CAJA", "CALIENTE", "CALOR", "CALZONCILLOS", "CAMA",
    "CAMARÓN", "CAMBIAR", "CAMINAR", "CAMIÓN/MANEJAR-CAMIÓN", "CAMISA", "CANADÁ", "CÁNCER", "CANSAR", "CARA", "CARÁCTER",
    "CARIBE", "CARNE", "CARRERA (ACADÉMICA)", "CARRO/MANEJAR-CARRO", "CARTA", "CARTAGO", "CASA", "CASTIGAR", "CATÓLICO", "CATORCE",
    "CEBOLLA", "CELESTE", "CELOS/CELOSO", "CENAREC", "CENTRO", "CENTROAMÉRICA", "CERCA", "CERDO", "CERO", "CERRAR",
    "CERTIFICADO", "CERVEZA", "CHILE (FRUTO)", "CHILE (PAÍS)", "CHINA", "CHINO", "CHOFER", "CIEGO", "CIEN", "CIEN-2",
    "CIENCIAS", "CINCO", "CINCUENTA", "CIUDAD-QUESADA", "CLARO", "CLASE", "CLIMA", "CLÍNICA", "COCA-COLA", "COCINA",
    "COCINAR", "COCODRILO", "COLABORAR", "COLOMBIA", "COLÓN", "COLOR", "COMA", "COMER", "CÓMO", "CÓMO-2",
    "CÓMODO", "COMPAÑERO", "COMPARAR", "COMPARTIR", "COMPLICAR", "COMPRAR", "COMPUTADORA", "COMUNICARSE/COMUNICACIÓN", "COMUNIDAD", "CON-CIERTA-FRECUENCIA",
    "CON-EL-TIEMPO", "CON-GUSTO", "CONOCER", "CONSOLAR", "CONSTRUIR", "CONTACTAR", "CONTAR", "CONTINUAR/SEGUIR", "CONTRA", "CONTRATAR",
    "CONTROLAR", "COPIAR", "CORONADO", "CORRECTO", "CORREO-ELECTRÓNICO", "CORRER", "CORRER-2", "CORTAR-RELACIÓN", "CORTO/BREVE", "COSTA-RICA",
    "CRECER", "CREER", "CRUZ-ROJA", "CUÁL", "CUÁL-2", "CUALQUIERA/NO-IMPORTAR", "CUANDO", "CUÁNTO", "CUARENTA", "CUARTO",
    "CUATRO", "CUATRO-ADJETIVO", "CUATROCIENTOS", "CUBA", "CUCARACHA", "CUCHARA", "CUCHILLO", "CUERPO", "CUIDAR-2", "CULANTRO",
    "CULTURA", "CUMPLEAÑOS", "CUMPLIR", "CUÑADO", "CURIOSIDAD/CURIOSO", "CURRIDABAT", "CURSO", "DAR", "DARSE-CUENTA", "DE-LEJOS",
    "DE-PIE", "DEBAJO", "DEBER", "DÉBIL", "DECIDIR", "DÉCIMO", "DECIR", "DECIR-2", "DEFECAR", "DEJAR",
    "DEJAR-2", "DELETREAR", "DELFÍN", "DEMANDAR/DENUNCIAR", "DENTRO", "DEPENDER", "DEPORTE", "DERECHO", "DESAMPARADOS", "DESAPARECER",
    "DESAPARECER-2", "DESARROLLAR", "DESARROLLAR-2", "DESCANSAR", "DESDE", "DESDE-ANTES-HASTA-AHORA", "DESDE-ANTES-HASTA-CIERTO-MOMENTO", "DESDE-CIERTO-MOMENTO-HACIA-ADELANTE", "DESDE-CIERTO-MOMENTO-HACIA-ATRÁS", "DESMORONAR",
    "DESORDEN", "DESPACIO", "DESPEDIR", "DESPEGAR-AVIÓN", "DESPERTARSE", "DESPLAZARSE", "DESPUÉS", "DESTRUIR", "DÍA", "DIABLO",
    "DIBUJAR", "DICCIONARIO", "DICIEMBRE", "DIECINUEVE", "DIECIOCHO", "DIECISÉIS", "DIECISIETE", "DIEZ", "DIFERENTE/SER-DIFERENTE", "DIFÍCIL",
    "DIMINUTO", "DIOS", "DIPUTADO", "DIRECCIÓN", "DIRECTOR", "DISCAPACIDAD", "DISCO", "DISCRIMINAR", "DISCUTIR", "DISEÑAR",
    "DISFRUTAR/DIVERTIRSE", "DISIMULAR", "DISMINUIR", "DIVISIÓN/DIVIDIR", "DIVORCIO", "DOCE", "DOCTOR", "DÓLAR", "DOLOR", "DOMINGO",
    "DÓNDE", "DORMIR", "DOS", "DOS-ADJETIVO", "DOSCIENTOS", "DUDAR", "DULCE (CARÁCTER)", "DULCE (SABOR)", "DURANTE-ESTE-TIEMPO", "DURAR",
    "DURO/ESTRICTO", "ECONOMÍA", "ECUADOR", "EDITAR", "EDUCACIÓN", "EJEMPLO", "EL-SALVADOR", "ELECTRICIDAD", "ELEFANTE", "EMOCIONARSE",
    "EMPATAR", "EMPEZAR/COMENZAR", "EMPLEADO", "EMPRESA/FÁBRICA", "EMPUJAR", "EN-LA-MAÑANA", "EN-LA-TARDE", "EN-SEGUNDO-LUGAR", "EN-SEGUNDO-LUGAR-2", "EN-TERCER-LUGAR-2",
    "ENAMORARSE", "ENAMORARSE-2", "ENCAJAR", "ENCONTRARSE-CON", "ENERO", "ENFATIZAR", "ENFERMO", "ENFRENTARSE", "ENGORDAR", "ENSALADA",
    "ENSEÑAR", "ENTENDER", "ENTONCES", "ENTRAR", "ENTRE-COMILLAS", "ENTRE-OTROS", "ENTREGAR-RECIBIR", "ENTRENADOR", "ENTREVISTAR", "ENVIAR",
    "EQUILIBRAR", "EQUIPO", "ERROR/EQUIVOCARSE", "ESCAZÚ", "ESCLAVO", "ESCOBILLAS", "ESCOGER", "ESCONDER/ESCONDIDO", "ESCONDERSE", "ESCRIBIR",
    "ESCRIBIR-EN-COMPUTADORA", "ESCUCHAR", "ESCUELA", "ESFORZAR", "ESO", "ESPAÑA", "ESPAÑOL", "ESPECIAL", "ESPERAR", "ESPERE!",
    "ESPÍRITU", "ESPOSO", "ESPOSO-2", "ESQUINA", "ESTABLECER", "ESTADIO", "ESTADOS-UNIDOS", "ESTAR-BALANCEADO", "ESTAR-BIEN", "ESTAR-BIEN-2",
    "ESTAR-CONFUNDIDO", "ESTAR-ESTRECHO-ECONÓMICAMENTE", "ESTAR-PRESENTE-DANDO-LA-CARA", "ESTAR-SATISFECHO", "ESTAR-TENTADO", "ESTATUTO", "ESTE", "ESTÓMAGO", "ESTUDIAR", "ESTUDIOS-SOCIALES",
    "EUROPA", "EXAMEN", "EXCELENTE", "EXCELENTE-2", "EXPANDIRSE", "EXPEDIENTE", "EXPERIMENTAR", "EXPLICAR", "EXPLOTAR", "EXPONER",
    "EXPRESAR", "EXPULSAR", "EXTENSIÓN-ESPACIO", "FACEBOOK", "FÁCIL", "FACTURA", "FALTAR", "FALTAR-AIRE", "FAMILIA", "FAMOSO",
    "FEBRERO", "FELICITAR", "FELIZ/CONTENTO/ALEGRE", "FEO", "FIEL", "FIESTA", "FILA", "FINCA", "FIRMAR", "FISCAL",
    "FÍSICA", "FLEXIBLE", "FLOR", "FLUIDO", "FORMA", "FORMARSE/SURGIR", "FORMATO", "FOTO", "FRACCIÓN", "FRANCIA",
    "FRASE", "FRENTE(CUERPO)", "FRESCO", "FRÍO", "FRUSTRAR", "FRUTA", "FRUTA-2", "FUERTE", "FUNCIÓN", "FUNDAR",
    "FUTURO", "GALLETA", "GALLINA", "GANAR", "GASOLINA", "GASTAR", "GATO", "GESTO", "GIMNASIO", "GOBIERNO",
    "GOL", "GOLFITO", "GOLPEAR", "GOLPEARSE", "GORDO", "GORILA", "GRABAR-SONIDO", "GRACIAS", "GRACIAS-A-DIOS", "GRADO",
    "GRADUARSE", "GRAMÁTICA", "GRANDE", "GRANDE-2", "GRATIS", "GRECIA", "GRIPE", "GRIS", "GRITAR", "GRÚA",
    "GRUPO", "GUADALUPE", "GUANACASTE", "GUANTES", "GUÁPILES", "GUARDAR", "GUATEMALA", "GUATUSO", "GUERRA", "GUÍA/GUIAR",
    "GUSTAR", "HABER/HAY", "HABLAR", "HACER-A-UN-LADO", "HAITÍ", "HARTO", "HASTA", "HATILLO", "HAY-DIFERENTES", "HELADO",
    "HELICÓPTERO", "HEREDIA", "HERMANO", "HÍGADO", "HIJA", "HIJO", "HIJO-2", "HIJUEPUTA", "HISTORIA", "HOJA (PLANTA)",
    "HOLA", "HOMBRE", "HONDURAS", "HONESTO", "HORARIO", "HOSPITAL", "HOTEL", "HOY", "HUELGA", "HUEVO",
    "HUMILDE", "HUNDIRSE/DECRECER", "IDENTIDAD", "IDIOMA", "IGNÓRELO!", "IMAGEN", "IMAGINAR", "IMPLANTE-COCLEAR", "IMPORTANTE", "IMPOTENTE",
    "IMPRIMIR", "INCLUIR", "INDEPENDIZAR", "INDIA", "INDIO", "INFORMAR", "INFORMAR-2", "INGENIERO", "INGLATERRA", "INGLÉS",
    "INOCENTE", "INSISTIR", "INSTITUTO", "INTEGRAR/INTEGRACIÓN", "INTELIGENTE", "INTERACTUAR", "INTERCAMBIAR", "INTERESANTE", "INTERNACIONAL", "INTÉRPRETE",
    "INTERRUMPIR", "INVENTAR", "INVESTIGAR", "INVIERNO", "INVITAR", "INYECTAR", "IR", "IR-2", "IR-3", "IRSE",
    "IRSE-2", "IRSE-3", "ISLA", "ITALIA", "JACÓ", "JAMAICA", "JAMÁS", "JAPÓN", "JESÚS", "JIRAFA",
    "JOVEN", "JUEVES", "JUGADOR", "JUGAR", "JULIO", "JUNIO", "JUNTOS/CON", "KINDER", "KINDER-2", "LA-FORTUNA",
    "LA-SABANA", "LÁPIZ", "LÁSTIMA", "LATINOAMÉRICA", "LECCIÓN", "LECHE", "LECTURA", "LEER", "LENGUA", "LEÓN",
    "LESCO", "LETRA", "LETRA-A", "LETRA-B", "LETRA-C", "LETRA-CH", "LETRA-D", "LETRA-E", "LETRA-F", "LETRA-G",
    "LETRA-H", "LETRA-I", "LETRA-J", "LETRA-K", "LETRA-L", "LETRA-LL", "LETRA-M", "LETRA-N", "LETRA-Ñ", "LETRA-O",
    "LETRA-P", "LETRA-Q", "LETRA-R", "LETRA-RR", "LETRA-S", "LETRA-T", "LETRA-U", "LETRA-V", "LETRA-W", "LETRA-X",
    "LETRA-Y", "LETRA-Z", "LEY", "LIBERIA", "LIBRE", "LIBRO", "LIMITADO", "LIMÓN (FRUTA)", "LIMÓN (LUGAR)", "LÍNEA", "LISTA", "LISTO",
    "LLAMAR-POR-TELÉFONO", "LLAVE", "LLEVAR-TRAER", "LLORAR", "LLOVER", "LOCO", "LÓGICO", "LUCHAR", "LUGAR", "LUNA",
    "LUNES", "LUNES-A-VIERNES", "LUZ", "MACARRONES", "MADERA", "MADRUGADA", "MADURAR/MADUREZ", "MAL", "MAL-2/ESTAR-MAL", "MAMÁ",
    "MAÑANA", "MANO", "MANO-2", "MÁQUINA", "MAR", "MARCHA", "MAREO", "MARTES", "MARTILLO", "MARZO",
    "MÁS", "MÁS-ADELANTE", "MÁS-O-MENOS", "MÁS-SUMA", "MATAR", "MATEMÁTICAS", "MAYO", "MAYORÍA", "MEDIAS", "MEDICINA",
    "MEJOR", "MEMORIA", "MENSAJE-DE-TEXTO", "MENTIR", "MERCADO-CENTRAL", "MES", "MESA", "MÉXICO", "MICROONDAS", "MIEDO",
    "MIÉRCOLES", "MIL", "MILLÓN", "MÍNIMO", "MINISTERIO/MINISTRO", "MINUTO", "MITAD/MEDIA", "MODELO", "MOJADO", "MONJA",
    "MONJE", "MONTAÑA", "MORADO", "MORAVIA", "MORENO", "MORIR", "MOSTRAR", "MOTIVAR", "MUCHO", "MUDARSE",
    "MUDO", "MUJER", "MUJER-2", "MULTIPLICAR", "MUNDO", "MUNDO-2", "MUSULMÁN", "MUY-POCO", "NACER", "NADA",
    "NADA-MÁS", "NADAR", "NARANJA", "NARIZ", "NECESITAR", "NEGRO", "NEGRO (ETNIA)", "NERVIOSO", "NICARAGUA", "NICOYA",
    "NIÑO", "NIÑO-2", "NO", "NO-2", "NO-3", "NO-4", "NO-ENTENDER-NADA", "NO-HABER-CAMPO", "NO-HAY/NO-HABER", "NO-PODER",
    "NO-PODER-DORMIR", "NO-SABER-QUÉ-HACER", "NO-SÉ/NO-SABER", "NOCHE", "NOMBRE", "NORMAL", "NORTE", "NOSOTROS-INDEF-1", "NOSOTROS-INDEF-2", "NOTA",
    "NOTICIA", "NOVECIENTOS", "NOVENO", "NOVENTA", "NOVIEMBRE", "NOVIO", "NUEVE", "NUEVO", "NÚMERO", "NUNCA",
    "O", "OBJETIVO", "OBLIGAR", "OCHENTA", "OCHO", "OCHOCIENTOS", "OCTAVO", "OCTUBRE", "OCTUBRE Dos", "OCUPADO",
    "ODIAR", "OESTE", "OFICIAL", "OFICINA", "OJALÁ", "OK", "ONCE", "OPINAR", "ORALIZAR", "ORINAR",
    "OSCURO", "OTRA-VEZ", "OTRO", "OVNI", "OYENTE", "PACIENCIA/AGUANTAR", "PADRES", "PADRES-2", "PAGAR", "PÁGINA",
    "PAÍS", "PÁJARO", "PALABRA", "PALMARES", "PALMARES-2", "PAN", "PANAMÁ", "PAPÁ", "PAPAS", "PARA",
    "PARA-QUÉ", "PARAGUAS", "PARAGUAY", "PARAÍSO", "PARÁLISIS-CEREBRAL", "PARAR-VEHÍCULO", "PARECER", "PARQUE", "PASADO", "PASEAR",
    "PATINES", "PATO-2", "PAYASO", "PAZ", "PECADO", "PEDIR", "PELIGROSO", "PENSAR", "PENSAR-2/ACORDARSE-DE", "PENSAR-3",
    "PEOR", "PEQUEÑO(ENTIDAD HORIZONTAL)", "PEQUEÑO(ENTIDAD VERTICAL)", "PERCIBIR", "PERDER", "PERDER/REPROBAR", "PEREZ-ZELEDÓN", "PERFECTO", "PERIÓDICO", "PERMISO",
    "PERRO", "PERSONA", "PERÚ", "PEZ", "PIES", "PIRÁMIDE", "PIZZA", "PLAYA", "PLAZA-DE-LA-CULTURA-1", "PLAZA-DE-LA-CULTURA-2",
    "PLAZA-VIQUEZ", "POBRE", "POCO", "POCO-A-POCO", "PODER", "POLLITO", "POLLO", "POLLO-2", "PONER", "POR-CIENTO/PORCENTAJE",
    "POR-ESO", "POR-FAVOR", "POR-SÍ-SOLO", "PORQUE", "POSTERGAR", "PRACTICAR", "PREFERIR", "PREGUNTAR", "PREMIO", "PREOCUPARSE",
    "PRESTAR", "PRIMERA-VEZ", "PRIMERO", "PRIMO", "PRIVADO", "PRO.POSESIVO", "PRO1", "PRO1.POSESIVO", "PRO2", "PROBAR",
    "PROBLEMA", "PRODUAL-2", "PRODUAL.1", "PROFESOR", "PROFUNDO", "PROMISMO", "PROPIO", "PROPLURAL", "PROPLURALDISTRIBUTIVO", "PROPONER",
    "PROTESTAR/QUEJARSE", "PROTRIAL.1", "PROVINCIA", "PSICOLOGÍA", "PUENTE", "PUERTO-RICO", "PULPERÍA", "PUNTARENAS", "QUE", "QUÉ-HACER?",
    "QUÉ-PASAR", "QUEDARSE", "QUEQUE", "QUERER", "QUIÉN", "QUÍMICA", "QUINCE", "QUINIENTOS", "QUINTO", "QUIZ",
    "RÁPIDO", "RARO", "RATÓN", "RAZÓN", "RECONOCER", "RECORDAR", "REDUCIRSE", "REGAÑAR", "REGAÑAR-DURO", "REGLA",
    "REINA", "REÍRSE", "RELIGIÓN", "RESOLVER", "RESPETAR", "RESPONDER", "RESPONSABLE", "RESTAR", "RESTAURANTE", "REY",
    "RÍO", "ROJO", "ROMANO", "ROSADO", "SÁBADO", "SABER", "SACERDOTE", "SALIR", "SAN-CARLOS", "SAN-JOSÉ",
    "SAN-PEDRO", "SAN-RAMÓN", "SANAR/SALUD", "SANTA-LUCÍA", "SANTO-DOMINGO", "SARCHI", "SECO", "SEGUIR-ADELANTE", "SEGUNDO", "SEGURO",
    "SEIS", "SEISCIENTOS", "SEMANA", "SEMBRAR", "SEMINARIO", "SEÑA", "SEÑA-2", "SEÑA-3", "SEÑA-4", "SENTARSE",
    "SENTIR", "SER-ALTO/ESTAR-GRANDE", "SER-IMPOSIBLE", "SER-PRO1", "SER-PRO2", "SER-TÍPICO-DE", "SERVIR", "SESENTA", "SETECIENTOS", "SETENTA",
    "SETIEMBRE", "SÉTIMO", "SEXO", "SEXTO", "SÍ", "SIEMPRE", "SIETE", "SIGNIFICAR", "SILLA", "SÍMBOLO",
    "SIMPLE", "SÍNDROME-DE-DOWN", "SITUACIÓN", "SOBRE", "SOBRINO", "SODA", "SOL", "SOLO", "SOPA", "SORDO",
    "SORDO^MUDO", "SORPRENDIDO", "SOSTÉN", "SUAVE", "SUBTÍTULOS", "SUCEDER/PASAR-ALGO", "SUCIO", "SUEÑO", "SUERTE", "SUFRIR",
    "SUMAR", "SUR", "TAL-VEZ", "TAMAL", "TAMBIÉN", "TAMBIÉN-2", "TAREA", "TAXI", "TÉ", "TÉ^FRIO",
    "TEMA", "TEMBLAR", "TENEDOR", "TENER", "TENER-CULPA", "TENER-EXPERIENCIA", "TENER-RAZÓN", "TENER-SUERTE", "TERCERO", "TERMINAR/ÚLTIMO",
    "TERRIBLE", "TIBÁS", "TIEMPO-2", "TIEMPO/HORA", "TIERRA(MATERIAL)", "TIERRA(PLANETA)", "TÍO", "TÍO-2", "TOCAR", "TODA-LA-MAÑANA",
    "TODAVÍA-NO", "TODO", "TODO-EL-DÍA", "TODOS-LOS-DOMINGOS", "TODOS-LOS-JUEVES", "TODOS-LOS-LUNES", "TODOS-LOS-MARTES", "TODOS-LOS-MIÉRCOLES", "TODOS-LOS-SÁBADOS", "TODOS-LOS-VIERNES",
    "TOMAR", "TOMATE", "TORTA", "TORTUGA", "TRABAJAR", "TRATAR", "TRATAR-DE-COMUNICARSE", "TRATAR-DE-CONVERSAR", "TRAUMA", "TRECE",
    "TREINTA", "TRES", "TRES-RÍOS", "TRESCIENTOS", "TURISMO", "TURQUÍA", "TURRIALBA", "TURRUCARES", "UNICO", "UNIVERSIDAD",
    "UNO", "UNO-MITAD", "URUGUAY", "USAR", "VACA/TORO", "VACACIONES", "VACÍO", "VAGABUNDO", "VAPOR", "VEGETAL",
    "VEINTE", "VENCER", "VENDER", "VENEZUELA", "VENIR", "VENTANA", "VER", "VER-2", "VER-3", "VERANO",
    "VERDAD", "VERDE", "VEZ", "VIDA", "VIEJO", "VIENTO", "VIERNES", "VIOLAR-LA-LEY", "VOCABULARIO", "VOLAR-EN-AVIÓN",
    "VOLCÁN", "VOZ", "WINDOWS", "WWW", "YA", "YA-2", "YUCA", "ZACATE", "ZAPATO", "ZAPOTE",
    "ZARCERO", "ZARCERO-2"]

    # Solicitar el directorio al usuario
    directorio = input("Introduce la ruta del directorio a analizar (Enter para el directorio actual): ").strip()
    if not directorio:
        directorio = "."

    # Ejecutar la detección
    faltantes = detectar_archivos_faltantes(directorio, nombres_esperados)
    if faltantes is None:
        return

    # Generar y mostrar el reporte
    generar_reporte(faltantes, len(nombres_esperados), directorio)

    # Preguntar si desea guardar la lista de archivos faltantes
    if faltantes:
        guardar = input("\n¿Deseas guardar la lista de archivos faltantes en un archivo? (s/n): ").lower()
        if guardar.startswith('s'):
            nombre_archivo = input("Nombre del archivo (presiona Enter para usar 'archivos_faltantes.txt'): ").strip()
            if not nombre_archivo:
                nombre_archivo = "archivos_faltantes.txt"
            guardar_lista_faltantes(faltantes, nombre_archivo)



def main():
    directory()
    
if __name__ == "__main__":
    main()
