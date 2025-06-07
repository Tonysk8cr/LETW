
import os
import difflib

# List of new names (without .mp4 extension)
NEW_NAMES = NEW_NAMES = [
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

def normalizar_nombre(nombre):
    """
    Devuelve una versión 'limpia' para comparación.
    Por ejemplo, quita sufijos como '-BROADBAND_LOW', 
    reemplaza guiones y subrayados por espacios,
    quita la extensión .mp4, etc.
    """
    # Quita la extensión .mp4 si existe
    if nombre.lower().endswith(".mp4"):
        nombre = nombre[:-4]  # elimina los últimos 4 caracteres

    # Reemplaza guiones y underscores por espacio
    nombre = nombre.replace("-", " ")
    nombre = nombre.replace("_", " ")
    nombre = nombre.replace("^", " ")
    nombre = nombre.replace("'", " ")
    nombre = nombre.replace("__", " ")
    nombre = nombre.replace("'UCR", " ")
    nombre = nombre.replace("'BROADBAND LOW", " ")
    nombre = nombre.replace("'Broadband_Low", " ")
    nombre = nombre.replace("-Broadband_Low", " ")

    # Elimina algunas frases específicas que no aportan a la comparación
    for basura in ["BROADBAND LOW", "CELLULAR LOW", "FORMANEUTRA", "(1)", "G", "Broadband_Low", "-Broadband_Low"]:
        nombre = nombre.upper().replace(basura.upper(), "")

    # Quita espacios dobles
    nombre = " ".join(nombre.split())

    # Opcional: podrías quitar acentos, tildes, etc. (si lo necesitas).
    # import unicodedata
    # nombre = ''.join((c for c in unicodedata.normalize('NFD', nombre) if unicodedata.category(c) != 'Mn'))

    return nombre.upper()

def renombrar_automatico(ruta_directorio, umbral=0.8):
    """
    Intenta renombrar cada archivo .mp4 del directorio buscándole
    la mejor coincidencia en NEW_NAMES, usando fuzzy matching.
    :param ruta_directorio: La carpeta donde están los archivos.
    :param umbral: Valor entre 0 y 1 que indica qué tan similar
                   debe ser para considerarla coincidencia aceptable.
    """
    # 1. Normaliza la lista de nombres "nuevos" y crea una tabla
    #    { normalizado: nombre_real }, para comparación rápida.
    map_new = {}
    for nuevo in NEW_NAMES:
        map_new[normalizar_nombre(nuevo)] = nuevo

    # 2. Verifica que la carpeta existe
    if not os.path.isdir(ruta_directorio):
        print(f"[ERROR] La carpeta '{ruta_directorio}' no existe.")
        return

    # 3. Cambia de directorio para simplificar los renombrados
    os.chdir(ruta_directorio)

    # 4. Obtén la lista de todos los archivos .mp4 en el directorio
    archivos = [archivo for archivo in os.listdir() if archivo.lower().endswith('.mp4')]
    print(f"[INFO] Se encontraron {len(archivos)} archivos .mp4 en el directorio")

    # Para mostrar progreso
    total = len(archivos)
    renombrados = 0
    sin_coincidencia = 0
    conflictos = 0
    errores = 0

    # 5. Recorre cada archivo en el directorio y busca coincidencia
    for i, old_name in enumerate(archivos):
        # Mostrar progreso
        if i % 10 == 0:
            print(f"[PROGRESO] Procesando archivo {i+1} de {total}...")

        # Normaliza el nombre viejo
        old_norm = normalizar_nombre(old_name)

        # Con get_close_matches buscamos en la lista "map_new.keys()"
        posibles = difflib.get_close_matches(old_norm, map_new.keys(), n=1, cutoff=umbral)
        if not posibles:
            print(f"[SIN COINCIDENCIA] '{old_name}' => (No se encontró nada similar)")
            sin_coincidencia += 1
            continue
        # Si tenemos una coincidencia, la tomamos
        mejor_coincidencia = posibles[0]  # un string "normalizado" del new
        new_base = map_new[mejor_coincidencia]  # recuperamos el "nombre real"
        new_name = new_base + ".mp4"            # le agregamos extensión

        # Intentamos renombrar
        if os.path.exists(new_name) and old_name != new_name:
            # Ya existe un archivo con ese nombre
            print(f"[CONFLICTO] Ya existe '{new_name}'. No se renombra '{old_name}'.")
            conflictos += 1
            continue

        try:
            if old_name != new_name:  # Evitar renombrar si ya tiene el nombre correcto
                os.rename(old_name, new_name)
                print(f"[OK] '{old_name}' => '{new_name}' (match: {old_norm} ~ {mejor_coincidencia})")
                renombrados += 1
            else:
                print(f"[SKIP] '{old_name}' ya tiene el nombre correcto")
        except Exception as e:
            print(f"[ERROR] Al renombrar '{old_name}' -> '{new_name}': {e}")
            errores += 1

    # 6. Mostrar resumen
    print("\n===== RESUMEN =====")
    print(f"Total de archivos procesados: {total}")
    print(f"Archivos renombrados exitosamente: {renombrados}")
    print(f"Archivos sin coincidencia: {sin_coincidencia}")
    print(f"Conflictos (archivo destino ya existe): {conflictos}")
    print(f"Errores durante el renombrado: {errores}")
    print("===================")


def mostrar_archivos_no_procesados(ruta_directorio):
    """
    Muestra los archivos .mp4 que no están en NEW_NAMES y los que no coinciden con ningún patrón.
    
    Args:
        ruta_directorio: Ruta donde están los archivos .mp4
    """
    try:
        # Verifica que la carpeta existe
        if not os.path.isdir(ruta_directorio):
            print(f"[ERROR] La carpeta '{ruta_directorio}' no existe.")
            return

        # Cambia al directorio
        os.chdir(ruta_directorio)

        # Obtiene todos los archivos .mp4
        archivos = [archivo for archivo in os.listdir() if archivo.lower().endswith('.mp4')]
        
        # Normaliza los nombres de NEW_NAMES
        nombres_validos = {normalizar_nombre(nombre) for nombre in NEW_NAMES}
        
        # Archivos que no están en NEW_NAMES
        no_en_lista = []
        no_coinciden = []

        for archivo in archivos:
            nombre_norm = normalizar_nombre(archivo)
            if nombre_norm not in nombres_validos:
                no_en_lista.append(archivo)
                # Verifica si hay alguna coincidencia cercana
                coincidencias = difflib.get_close_matches(nombre_norm, nombres_validos, n=1, cutoff=0.8)
                if not coincidencias:
                    no_coinciden.append(archivo)

        # Muestra resultados
        print("\n===== ARCHIVOS NO PROCESADOS =====")
        print(f"\nArchivos no encontrados en la lista ({len(no_en_lista)}):")
        for archivo in no_en_lista:
            print(f"- {archivo}")
            
        print(f"\nArchivos sin coincidencias cercanas ({len(no_coinciden)}):")
        for archivo in no_coinciden:
            print(f"- {archivo}")
        
        print("\n===============================")
        
    except Exception as e:
        print(f"Error: {e}")

# Ejemplo de uso:
# mostrar_archivos_no_procesados(r"C:\Users\tonyi\Documents\LETW\T_Videos")

if __name__ == "__main__":
    # Cambia esta ruta según donde tengas tus archivos:
    carpeta_videos = input("Ingrese la ruta del directorio (presione Enter para usar la ruta por defecto): ") or r"C:\Users\tonyi\Documents\LETW\T_Videos"
    
    # Si la ruta está vacía, usar el directorio actual
    if not carpeta_videos.strip():
        carpeta_videos = "."
        print("[INFO] Usando directorio actual.")
    
    # Preguntar por el umbral de coincidencia
    try:
        umbral = float(input("Introduce el umbral de coincidencia (0.0-1.0, recomendado 0.8, presiona Enter para usar 0.8): ") or "0.8")
    except ValueError:
        umbral = 0.8
        print("[INFO] Usando umbral predeterminado: 0.8")
    
    # Confirmar antes de proceder
    print(f"\nSe van a procesar todos los archivos .mp4 en '{carpeta_videos}' con umbral {umbral}")
    confirmacion = input("¿Continuar? (s/n): ").lower()
    
    if confirmacion == 's' or confirmacion == 'si' or confirmacion == 'sí' or confirmacion == 'y' or confirmacion == 'yes':
        # Llamada principal
        renombrar_automatico(carpeta_videos, umbral=umbral)
    else:
        print("Operación cancelada.")

    mostrar_archivos_no_procesados(r"C:\Users\tonyi\OneDrive\Documentos\LETW\T_Videos")
