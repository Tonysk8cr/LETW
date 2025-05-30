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
    actions = np.array([
        "A-PARTIR-DE.mp4", "A-VECES-2.mp4", "A-VECES.mp4", "ABOGADO.mp4", "ABRAZAR.mp4", "ABRIL.mp4", "ABRIR.mp4", "ABUELO.mp4", "ACCIDENTE.mp4", "ACOMPAÑAR.mp4",
    "ACONGOJARSE.mp4", "ACONSEJAR.mp4", "ACOSTUMBRAR.mp4", "ACTIVIDAD.mp4", "ACUERDO.mp4", "ADELANTAR.mp4", "ADJUNTAR.mp4", "ADULTO.mp4", "AFILIAR.mp4", "AFRICA.mp4",
    "AGOSTO.mp4", "AGREGAR.mp4", "AGRUPARSE.mp4", "AHORA.mp4", "AHORITA.mp4", "AHORRAR.mp4", "AISLADO-EN-CASA.mp4", "AL-FRENTE.mp4", "ALAJUELA.mp4", "ALAJUELITA.mp4",
    "ALBANIA.mp4", "ALEMANIA.mp4", "ALGO.mp4", "ALIVIO.mp4", "ALMA.mp4", "ALQUILER.mp4", "AMARILLO.mp4", "AMBOS.mp4", "AMETRALLAR.mp4", "AMÉRICA.mp4",
    "ANALIZAR.mp4", "ANARANJADO.mp4", "ANASCOR.mp4", "ANGEL.mp4", "ANIMAL.mp4", "ANTES.mp4", "APARECER.mp4", "APARIENCIA.mp4", "APARTAR (2).mp4", "APARTAR.mp4",
    "APELLIDO.mp4", "APESTAR.mp4", "APLAUDIR.mp4", "APLICAR.mp4", "APOYAR.mp4", "APRENDER.mp4", "APROBAR.mp4", "APROVECHAR.mp4", "ARABIA-SAUDÍ.mp4", "ARABIA.mp4",
    "ARBITRO.mp4", "ARBOL.mp4", "ARREGLAR.mp4", "ARREPENTIRSE.mp4", "ASAMBLEA-LEGISLATIVA.mp4", "ASAMBLEA.mp4", "ASERRÍ.mp4", "ASIA.mp4", "ASOCIACIÓN.mp4", "ASOCIADO.mp4",
    "ASUSTAR.mp4", "ATENDER.mp4", "ATERRIZAR.mp4", "ATRASAR.mp4", "AUDÍFONO.mp4", "AVISAR.mp4", "AYER.mp4", "AYUDAR.mp4", "AZUL.mp4", "AÑO.mp4",
    "BACHILLERATO.mp4", "BAILAR.mp4", "BALLENA-2.mp4", "BALLENA.mp4", "BANANO.mp4", "BANCO (2).mp4", "BANCO.mp4", "BARATO.mp4", "BARVA.mp4", "BASURA.mp4",
    "BAÑARSE.mp4", "BAÑO.mp4", "BEBÉ.mp4", "BENEFICIARSE.mp4", "BIBLIA.mp4", "BICICLETA.mp4", "BIEN.mp4", "BIENVENIDO.mp4", "BILINGÜISMO.mp4", "BIOLOGÍA.mp4",
    "BLANCO.mp4", "BODA.mp4", "BOLIVIA.mp4", "BONITO.mp4", "BOSQUE.mp4", "BOSTON.mp4", "BRASIL.mp4", "BRIBRI.mp4", "BRILLAR.mp4", "BRINCAR.mp4",
    "BROMA.mp4", "BRUTO.mp4", "BUENO.mp4", "BUS.mp4", "BUSCAR.mp4", "CABALLO.mp4", "CADA-UNO.mp4", "CAERSE-2.mp4", "CAERSE.mp4", "CAFÉ (BEBIDA).mp4",
    "CAFÉ (COLOR).mp4", "CAJA.mp4", "CALIENTE.mp4", "CALOR.mp4", "CALZONCILLOS.mp4", "CAMA.mp4", "CAMARÓN.mp4", "CAMBIAR.mp4", "CAMINAR.mp4", "CAMISA.mp4",
    "CANADÁ.mp4", "CANSAR.mp4", "CARA.mp4", "CARIBE.mp4", "CARNE.mp4", "CARTA.mp4", "CARTAGO.mp4", "CARÁCTER.mp4", "CASA.mp4", "CASTIGAR.mp4",
    "CATÓLICO.mp4", "CAtorce.mp4", "CEBOLLA.mp4", "CELESTE.mp4", "CELOS.mp4", "CENAREC.mp4", "CENTROAMÉRICA.mp4", "CERCA.mp4", "CERDO.mp4", "CERO.mp4",
    "CERTIFICADO.mp4", "CERVEZA.mp4", "CHILE (PAÍS).mp4", "CHILE comida.mp4", "CHINA.mp4", "CHINO.mp4", "CHOFER.mp4", "CIEGO.mp4", "CIEN.mp4", "CIENCIAS.mp4",
    "CINCO.mp4", "CINCUENTA.mp4", "CIUDAD-QUESADA.mp4", "CLARO.mp4", "CLASE.mp4", "CLIMA.mp4", "CLÍNICA.mp4", "COCA-COLA.mp4", "COCINA.mp4", "COCINAR.mp4",
    "COCODRILO.mp4", "COLABORAR.mp4", "COLOMBIA.mp4", "COLOR.mp4", "COLÓN.mp4", "COMA.mp4", "COMER.mp4", "COMPARAR.mp4", "COMPARTIR.mp4", "COMPAÑERO.mp4",
    "COMPLICAR.mp4", "COMPRAR.mp4", "COMPUTADORA.mp4", "COMUNICAR.mp4", "COMUNIDAD.mp4", "CON-CIERTA-FRECUENCIA.mp4", "CON-EL-TIEMPO.mp4", "CON-GUSTO.mp4", "CONOCER.mp4", "CONSOLAR.mp4",
    "CONSTRUIR.mp4", "CONTACTAR.mp4", "CONTAR.mp4", "CONTRATAR.mp4", "CONTROLAR.mp4", "COPIAR.mp4", "CORONADO.mp4", "CORRECTO.mp4", "CORRER 2.mp4", "CORRER-2.mp4",
    "CORRER.mp4", "CORRTO.mp4", "CORTAR.mp4", "COSTA-RICA.mp4", "CRECER.mp4", "CREER.mp4", "CRUZ-ROJA.mp4", "CUANDO.mp4", "CUARENTA.mp4", "CUATRO-ADJETIVO.mp4",
    "CUATRO.mp4", "CUATROCIENTOS.mp4", "CUBA.mp4", "CUCARACHA.mp4", "CUCHARA.mp4", "CUCHILLO.mp4", "CUERPO.mp4", "CUIDAR-2.mp4", "CULANTRO (2).mp4", "CULANTRO.mp4",
    "CULTURA.mp4", "CUMPLEAÑOS.mp4", "CUMPLIR.mp4", "CURIOSIDAD.mp4", "CURRIDABAT.mp4", "CURSO.mp4", "CUÁL-2.mp4", "CUÁL.mp4", "CUÑADO.mp4", "Camión.mp4",
    "Cualquiera.mp4", "CÓMO-2.mp4", "CÓMODO.mp4", "DAR.mp4", "DARSE-CUENTA.mp4", "DE-LEJOS.mp4", "DE-PIE.mp4", "DEBAJO.mp4", "DEBER.mp4", "DECIDIR.mp4",
    "DECIR-2.mp4", "DECIR.mp4", "DECRECER.mp4", "DEFECAR.mp4", "DEJAR-2.mp4", "DELETREAR.mp4", "DELFÍN.mp4", "DENTRO.mp4", "DEPENDER.mp4", "DEPORTE.mp4",
    "DERECHO.mp4", "DESAMPARADOS.mp4", "DESAPARECER.mp4", "DESARROLLAR-2.mp4", "DESARROLLAR.mp4", "DESCANSAR.mp4", "DESDE-ANTES-HASTA-AHORA.mp4", "DESDE-ANTES-HASTA-CIERTO-MOMENTO.mp4", "DESDE-CIERTO-MOMENTO-HACIA-ADELANTE.mp4", "DESDE.mp4",
    "DESMORONAR.mp4", "DESORDEN.mp4", "DESPACIO.mp4", "DESPEDIR.mp4", "DESPERTARSE.mp4", "DESPLAZARSE.mp4", "DESPUÉS.mp4", "DESTRUIR.mp4", "DES^PARCER.mp4", "DIABLO.mp4",
    "DIBUJAR.mp4", "DICCIONARIO.mp4", "DICIEMBRE.mp4", "DIECINUEVE.mp4", "DIECIOCHO.mp4", "DIECISIETE.mp4", "DIECISÉIS.mp4", "DIEZ.mp4", "DIFÍCIL.mp4", "DIMINUTO.mp4",
    "DIOS (2).mp4", "DIOS.mp4", "DIPUTADO.mp4", "DIRECCIÓN.mp4", "DIRECTOR.mp4", "DISCAPACIDAD.mp4", "DISCO.mp4", "DISCRIMINAR.mp4", "DISCUTIR.mp4", "DISEÑAR.mp4",
    "DISIMULAR.mp4", "DISMINUIR.mp4", "DIVISION.mp4", "DIVORCIO.mp4", "DOCE.mp4", "DOLOR.mp4", "DOMINGO.mp4", "DORMIR.mp4", "DOS (2).mp4", "DOS-ADJETIVO.mp4",
    "DOS.mp4", "DOSCIENTOS (2).mp4", "DOSCIENTOS.mp4", "DUDAR.mp4", "DULCE (CARÁCTER).mp4", "DULCE (SABOR).mp4", "DURAR.mp4", "DÉBIL.mp4", "DÉCIMO.mp4", "DÓLAR.mp4",
    "DÓNDE.mp4", "ECONOMÍA.mp4", "ECUADOR.mp4", "EDITAR.mp4", "EDUCACIÓN.mp4", "EJEMPLO.mp4", "EL-SALVADOR.mp4", "ELECTRICIDAD.mp4", "ELEFANTE.mp4", "EMOCIONARSE.mp4",
    "EMPATAR.mp4", "EMPEZAR.mp4", "EMPLEADO.mp4", "EMPRESA.mp4", "EMPUJAR.mp4", "EN-LA-MAÑANA.mp4", "EN-LA-TARDE.mp4", "EN-SEGUNDO-LUGAR-2.mp4", "EN-SEGUNDO-LUGAR.mp4", "EN-TERCER-LUGAR-2.mp4",
    "ENAMORARSE-2.mp4", "ENAMORARSE.mp4", "ENCAJAR.mp4", "ENCONTRARSE-CON.mp4", "ENERO.mp4", "ENFATIZAR.mp4", "ENFERMO.mp4", "ENGORDAR.mp4", "ENSALADA.mp4", "ENSEÑAR.mp4",
    "ENTENDER.mp4", "ENTONCES.mp4", "ENTRAR.mp4", "ENTRE-COMILLAS.mp4", "ENTRE-OTROS.mp4", "ENTREGAR.mp4", "ENTRENADOR.mp4", "ENTREVISTAR.mp4", "ENVIAR.mp4", "EQUILIBRAR.mp4",
    "EQUIVOCARSE.mp4", "ESCAZÚ.mp4", "ESCLAVO.mp4", "ESCOBILLAS.mp4", "ESCOGER.mp4", "ESCONDERSE (2).mp4", "ESCONDERSE.mp4", "ESCRIBIR.mp4", "ESCUCHAR.mp4", "ESCUELA.mp4",
    "ESFUERZO.mp4", "ESO.mp4", "ESPAÑA.mp4", "ESPAÑOL.mp4", "ESPECIAL.mp4", "ESPERAR (2).mp4", "ESPERAR.mp4", "ESPERE!.mp4", "ESPOSO-2.mp4", "ESPOSO.mp4",
    "ESPÍRITU.mp4", "ESQUINA (2).mp4", "ESQUINA.mp4", "ESTABLECER.mp4", "ESTADIO.mp4", "ESTADOS-UNIDOS.mp4", "ESTAR MAL.mp4", "ESTAR-BALANCEADO.mp4", "ESTAR-BIEN.mp4", "ESTAR-ESTRECHO-ECONÓMICAMENTE.mp4",
    "ESTAR-PRESENTE-DANDO-LA-CARA.mp4", "ESTAR-SATISFECHO.mp4", "ESTAR-TENTADO.mp4", "ESTATUTO.mp4", "ESTE.mp4", "ESTUDIAR.mp4", "ESTUDIOS-SOCIALES.mp4", "ESTÓMAGO.mp4", "EUROPA.mp4", "EXAMEN.mp4",
    "EXCELENTE.mp4", "EXPEDIENTE.mp4", "EXPERIMENTAR.mp4", "EXPLICAR.mp4", "EXPLOTAR.mp4", "EXPONER.mp4", "EXPRESAR.mp4", "EXPULSAR.mp4", "EXTENSIÓN-ESPACIO.mp4", "Enojarse.mp4",
    "FACEBOOK.mp4", "FACTURA.mp4", "FALTAR-AIRE.mp4", "FALTAR.mp4", "FAMILIA.mp4", "FAMOSO.mp4", "FEBRERO.mp4", "FELICITAR.mp4", "FELIZ.mp4", "FEO.mp4",
    "FIEL.mp4", "FIESTA.mp4", "FILA.mp4", "FINCA.mp4", "FIRMAR.mp4", "FISCAL.mp4", "FLEXIBLE.mp4", "FLOR.mp4", "FLUIDO.mp4", "FORMA.mp4",
    "FORMAR.mp4", "FORMATO.mp4", "FOTO.mp4", "FRACCIÓN.mp4", "FRANCIA.mp4", "FRASE.mp4", "FRENTE.mp4", "FRESCO.mp4", "FRUSTRAR (2).mp4", "FRUSTRAR.mp4",
    "FRUTA-2.mp4", "FRUTA.mp4", "FRÍO.mp4", "FUERTE.mp4", "FUNCIÓN.mp4", "FUNDAR.mp4", "FUTURO.mp4", "FÁCIL.mp4", "FÍSICA.mp4", "GALLETA.mp4",
    "GALLINA.mp4", "GANAR.mp4", "GASOLINA.mp4", "GASTAR.mp4", "GATO.mp4", "GESTO.mp4", "GIMNASIO.mp4", "GOBIERNO.mp4", "GOL.mp4", "GOLFITO.mp4",
    "GOLPE.mp4", "GOLPEAR.mp4", "GORDO.mp4", "GORILA.mp4", "GRABAR.mp4", "GRACIAS-A-DIOS.mp4", "GRACIAS.mp4", "GRADO.mp4", "GRADUARSE.mp4", "GRAMÁTICA.mp4",
    "GRANDE-2.mp4", "GRANDE.mp4", "GRATIS.mp4", "GRECIA.mp4", "GRIPE.mp4", "GRIS.mp4", "GRITAR.mp4", "GRUPO.mp4", "GUADALUPE.mp4", "GUANACASTE.mp4",
    "GUANTES (2).mp4", "GUANTES.mp4", "GUARDAR.mp4", "GUATEMALA.mp4", "GUATUSO.mp4", "GUERRA.mp4", "GUIAR.mp4", "GUSTAR.mp4", "GUÁPILES.mp4", "HABLAR.mp4",
    "HACER-A-UN-LADO.mp4", "HAITÍ.mp4", "HARTO.mp4", "HASTA.mp4", "HATILLO.mp4", "HAY-DIFERENTES.mp4", "HAY.mp4", "HELADO.mp4", "HELICÓPTERO.mp4", "HEREDIA.mp4",
    "HERMano.mp4", "HIJO-2.mp4", "HIJO2.mp4", "HIJUEPUTA.mp4", "HISTORIA.mp4", "HOLA.mp4", "HOLa (2).mp4", "HOMBRE.mp4", "HONDURAS.mp4", "HORARIO.mp4",
    "HOSPITAL.mp4", "HOTEL.mp4", "HOY.mp4", "HUELGA.mp4", "HUEVO.mp4", "HUMILDE.mp4", "HÍGADO.mp4", "IDENTIFICAR.mp4", "IDIOMA.mp4", "IMAGEN.mp4",
    "IMAGINAR.mp4", "IMPLANTE-COCLEAR.mp4", "IMPORTANTE.mp4", "IMPOTENTE.mp4", "IMPRIMIR.mp4", "INDEPENDIZAR.mp4", "INDIA.mp4", "INDIO.mp4", "INFORMAR.mp4", "INGENIERO.mp4",
    "INGLATERRA.mp4", "INGLÉS.mp4", "INOCENTE.mp4", "INSISTIR.mp4", "INSTITUTO.mp4", "INTELIGENTE.mp4", "INTERACTUAR.mp4", "INTERESANTE.mp4", "INTERNACIONAL.mp4", "INTERRUMPIR.mp4",
    "INTÉRPRETE.mp4", "INVENTAR.mp4", "INVIERNO.mp4", "INVITAR.mp4", "INYECTAR.mp4", "IR-2.mp4", "IR-3.mp4", "IR.mp4", "IRSE-2.mp4", "IRSE-3.mp4",
    "IRSE.mp4", "ISLA.mp4", "ITALIA.mp4", "JACÓ.mp4", "JAMAICA.mp4", "JAMÁS.mp4", "JAPÓN.mp4", "JESÚS.mp4", "JIRAFA.mp4", "JUEVES.mp4",
    "JUGADOR.mp4", "JUGAR.mp4", "JULIO.mp4", "JUNIO.mp4", "KINDER-2.mp4", "KINDER.mp4", "LA-SABANA.mp4", "LATINOAMÉRICA.mp4", "LECCIÓN.mp4", "LECHE.mp4",
    "LECTURA.mp4", "LEER.mp4", "LENGUA.mp4", "LESCO.mp4", "LETRA A.mp4", "LETRA G.mp4", "LETRA-A.mp4", "LETRA-B.mp4", "LETRA-C.mp4", "LETRA-CH.mp4",
    "LETRA-D.mp4", "LETRA-E.mp4", "LETRA-F.mp4", "LETRA-G.mp4", "LETRA-H.mp4", "LETRA-I.mp4", "LETRA-J.mp4", "LETRA-K.mp4", "LETRA-L.mp4", "LETRA-LL.mp4",
    "LETRA-M.mp4", "LETRA-N.mp4", "LETRA-O.mp4", "LETRA-P.mp4", "LETRA-Q.mp4", "LETRA-R.mp4", "LETRA-RR.mp4", "LETRA-S.mp4", "LETRA-T.mp4", "LETRA-U.mp4",
    "LETRA-V.mp4", "LETRA-W.mp4", "LETRA-X.mp4", "LETRA-Y.mp4", "LETRA-Z.mp4", "LETRA-Ñ.mp4", "LEY.mp4", "LIBERIA.mp4", "LIBRE.mp4", "LIBRO.mp4",
    "LIMITADO.mp4", "LIMÓN (FRUTA).mp4", "LIMÓN (LUGAR).mp4", "LISTA.mp4", "LISTO.mp4", "LLAMAR-POR-TELÉFONO.mp4", "LLAVE.mp4", "LLEVAR-TRAER.mp4", "LLORAR.mp4", "LUCHAR.mp4",
    "LUGAR.mp4", "LUNA.mp4", "LUNES-A-VIERNES.mp4", "LUNES.mp4", "LÁPIZ.mp4", "LÁSTIMA.mp4", "LÍNEA.mp4", "LÓGICO.mp4", "MADERA.mp4", "MADRUGADA.mp4",
    "MAL.mp4", "MAMÁ.mp4", "MANO-2.mp4", "MANO.mp4", "MARCHA.mp4", "MARTES.mp4", "MARTILLO.mp4", "MARZO.mp4", "MATAR.mp4", "MATEMÁTICAS.mp4",
    "MAYO.mp4", "MAYORÍA.mp4", "MAÑANA.mp4", "MEDIAS.mp4", "MEDICINA (2).mp4", "MEDICINA.mp4", "MEJOR.mp4", "MEMORIA.mp4", "MENSAJE-DE-TEXTO.mp4", "MENTIR.mp4",
     "MERCADO-CENTRAL.mp4", "MES.mp4", "MESA.mp4", "MICROONDAS.mp4", "MIEDO.mp4", "MIL (2).mp4", "MIL.mp4", "MILLÓN.mp4", "MINISTERIO.mp4", "MINUTO.mp4",
    "MITAD.mp4", "MIÉRCOLES.mp4", "MODELO.mp4", "MOJADO.mp4", "MONJA.mp4", "MONTAÑA.mp4", "MORADO (2).mp4", "MORADO.mp4", "MORAVIA.mp4", "MORENO.mp4",
    "MORIR.mp4", "MOSTRAR.mp4", "MOTIVAR.mp4", "MUCHO.mp4", "MUDARSE.mp4", "MUDO.mp4", "MUJER (2).mp4", "MUJER-2.mp4", "MUJER.mp4", "MULTIPLICAR.mp4",
    "MUNDO-2 (2).mp4", "MUNDO-2.mp4", "MUNDO.mp4", "MUSULMÁN.mp4", "MUY-POCO.mp4", "MÁS-ADELANTE.mp4", "MÁS-O-MENOS.mp4", "MÁS.mp4", "MÄS.mp4", "MÉXICO.mp4",
    "MÍNIMO.mp4", "NACER.mp4", "NADA-MÁS.mp4", "NADA.mp4", "NADAR.mp4", "NARANJA.mp4", "NARIZ.mp4", "NECESITAR.mp4", "NEGRO (ETNIA).mp4", "NEGRO.mp4",
    "NERVIOSO.mp4", "NICARAGUA.mp4", "NICOYA.mp4", "NIÑO-2.mp4", "NIÑO.mp4", "NO HAY.mp4", "NO SË.mp4", "NO-2.mp4", "NO-3.mp4", "NO-4.mp4",
    "NO-ENTENDER-NADA.mp4", "NO-HABER-CAMPO.mp4", "NO-PODER-DORMIR.mp4", "NO-PODER.mp4", "NO-SABER-QUÉ-HACER.mp4", "NOCHE.mp4", "NOMBRE.mp4", "NORMAL.mp4", "NORTE.mp4", "NOSOTROS-INDEF-1.mp4",
    "NOSOTROS-INDEF-2.mp4", "NOTA (2).mp4", "NOTA.mp4", "NOTICIA.mp4", "NOVECIENTOS.mp4", "NOVENO.mp4", "NOVENTA.mp4", "NOVIEMBRE.mp4", "NOVIO.mp4", "NUEVE.mp4",
    "NUEVO.mp4", "NUNCA.mp4", "NÚMERO.mp4", "O.mp4", "OBJETIVO.mp4", "OBLIGAR.mp4", "OCHENTA.mp4", "OCHO.mp4", "OCHOCIENTOS.mp4", "OCTUBRE-2.mp4",
    "OCTUBRE.mp4", "OCUPADO.mp4", "ODIAR.mp4", "OESTE.mp4", "OFICIAL.mp4", "OFICINA.mp4", "OJALÁ.mp4", "OK.mp4", "ONCE.mp4", "OPINAR.mp4",
    "ORALIZAR.mp4", "ORINAR.mp4", "OSCURO.mp4", "OTRA-VEZ.mp4", "OTRO.mp4", "OVNI.mp4", "OYENTE.mp4", "PACIENCIA.mp4", "PADRES (2).mp4", "PADRES.mp4",
    "PALABRA.mp4", "PALMARES-2 (2).mp4", "PALMARES-2.mp4", "PALMARES.mp4", "PAN.mp4", "PANAMÁ.mp4", "PAPAS (2).mp4", "PAPAS.mp4", "PARA-QUÉ.mp4", "PARAGUAS.mp4",
    "PARAGUAY.mp4", "PARAÍSO.mp4", "PARECER.mp4", "PARQUE.mp4", "PARÁLISIS-CEREBRAL.mp4", "PASADO.mp4", "PASEAR.mp4", "PATINES.mp4", "PAYASO.mp4", "PAZ.mp4",
    "PECADO.mp4", "PEDIR.mp4", "PELIGROSO.mp4", "PENSAR-3.mp4", "PENSAR.mp4", "PEOR.mp4", "PEQUEÑO-2.mp4", "PEQUEÑO.mp4", "PERCIBIR.mp4", "PERDER-REPROBAR.mp4",
    "PERDER.mp4", "PEREZ-ZELEDÓN.mp4", "PERFECTO.mp4", "PERIÓDICO.mp4", "PERRO.mp4", "PERSONA.mp4", "PEZ.mp4", "PIRÁMIDE.mp4", "PIZZA.mp4", "PLAZA-DE-LA-CULTURA-1.mp4",
    "PLAZA-DE-LA-CULTURA-2.mp4", "PLAZA-VIQUEZ.mp4", "POCO-A-POCO.mp4", "POCO.mp4", "PODER.mp4", "POLLITO.mp4", "POLLO-2.mp4", "POLLO.mp4", "PONER.mp4", "POR-ESO (2).mp4",
    "POR-ESO.mp4", "POR-FAVOR.mp4", "POR-SÍ-SOLO.mp4", "PORQUE.mp4", "POSTERGAR.mp4", "PRACTICAR.mp4", "PREFERIR.mp4", "PREGUNTAR.mp4", "PREMIO.mp4", "PREOCUPARSE.mp4",
    "PRESTAR.mp4", "PRIMERA-VEZ.mp4", "PRIMERO.mp4", "PRIMO.mp4", "PRIVADO.mp4", "PRO1.POSESIVO.mp4", "PRO1.mp4", "PROBAR.mp4", "PROBLEMA.mp4", "PRODUAL-2.mp4",
    "PRODUAL.1.mp4", "PROFESOR.mp4", "PROFUNDO.mp4", "PROMISMO.mp4", "PROPIO (2).mp4", "PROPIO.mp4", "PROPLURAL.mp4", "PROPLURALDISTRIBUTIVO.mp4", "PROPONER.mp4", "PROTRIAL.1.mp4",
    "PROVINCIA.mp4", "PSICOLOGÍA.mp4", "PUENTE.mp4", "PUERTO-RICO.mp4", "PULPERÍA.mp4", "PUNTARENAS.mp4", "PÁGINA.mp4", "PÁJARO.mp4", "QUE.mp4", "QUEDARSE.mp4",
    "QUEQUE.mp4", "QUERER.mp4", "QUINCE.mp4", "QUINIENTOS.mp4", "QUIZ.mp4", "QUIÉN.mp4", "QUÉ-HACER.mp4", "QUÉ-PASAR.mp4", "QUÍMICA.mp4", "RARO.mp4",
    "RATÓN.mp4", "RAZÓN.mp4", "RECONOCER.mp4", "RECORDAR.mp4", "REDUCIRSE.mp4", "REGAÑAR-DURO.mp4", "REGAÑAR.mp4", "REGLA.mp4", "REINA.mp4", "RELIGIÓN.mp4",
    "RESOLVER.mp4", "RESPETAR.mp4", "RESPONDER.mp4", "RESPONSABLE.mp4", "RESTAR.mp4", "REY.mp4", "REÍRSE.mp4", "ROJO.mp4", "ROMANO.mp4", "ROSADO.mp4",
    "RÍO.mp4", "SABER.mp4", "SACERDOTE.mp4", "SALIR.mp4", "SAN-CARLOS.mp4", "SAN-JOSÉ.mp4", "SAN-PEDRO.mp4", "SAN-RAMÓN.mp4", "SANAR.mp4", "SANTA-LUCÍA.mp4",
    "SANTO-DOMINGO.mp4", "SARCHI.mp4", "SECO.mp4", "SEIS.mp4", "SEISCIENTOS.mp4", "SEMANA.mp4", "SEMBRAR.mp4", "SEMINARIO.mp4", "SENTARSE.mp4", "SENTIR.mp4",
    "SER DIFERENTE.mp4", "SER-ALTO.mp4", "SER-IMPOSIBLE.mp4", "SER-PRO1.mp4", "SER-PRO2.mp4", "SER-TÍPICO-DE.mp4", "SERVIR.mp4", "SESENTA.mp4", "SETECIENTOS.mp4", "SETENTA.mp4",
    "SETIEMBRE.mp4", "SEXO.mp4", "SEÑA-4.mp4", "SIEMPRE.mp4", "SIETE.mp4", "SIGNIFICAR.mp4", "SILLA.mp4", "SIMPLE.mp4", "SITUACIÓN.mp4", "SOBRE.mp4",
    "SOBRINO.mp4", "SODA.mp4", "SOLO.mp4", "SOPA.mp4", "SORDO.mp4", "SORDO^MUDO.mp4", "SORPRENDIDO.mp4", "SOSTÉN.mp4", "SUAVE.mp4", "SUBTÍTULOS.mp4",
    "SUCEDER.mp4", "SUCIO.mp4", "SUFRIR.mp4", "SUMAR.mp4", "Suerte.mp4", "Suerte2.mp4", "Sur.mp4", "SÁBADO.mp4", "SÉTIMO.mp4", "SÍ.mp4",
    "SÍMBOLO.mp4", "SÍNDROME-DE-DOWN.mp4", "TAL-VEZ.mp4", "TAMAL.mp4", "TAMBIÉN-2.mp4", "TAMBIÉN.mp4", "TENER-EXPERIENCIA.mp4", "TENER-RAZÓN.mp4", "TENER-SUERTE.mp4", "TEma.mp4",
    "TIBÁS.mp4", "TIEMPO-2.mp4", "TIERRA(PLANETA).mp4", "TODA-LA-MAÑANA.mp4", "TODAVÍA-NO.mp4", "TODO-EL-DÍA.mp4", "TODOS-LOS-DOMINGOS.mp4", "TODOS-LOS-JUEVES.mp4", "TODOS-LOS-LUNES.mp4", "TODOS-LOS-MARTES.mp4",
    "TODOS-LOS-MIÉRCOLES.mp4", "TODOS-LOS-SÁBADOS.mp4", "TODOS-LOS-VIERNES.mp4", "TRABAJAR.mp4", "TRATAR-DE-COMUNICARSE.mp4", "TRATAR-DE-CONVERSAR.mp4", "TRATAR.mp4", "TRES-RÍOS.mp4", "TRES.mp4", "TRESCIENTOS.mp4",
    "TURQUÍA.mp4", "Tarea.mp4", "Temblar.mp4", "Tenedor.mp4", "Tener-Culpa.mp4", "Tener.mp4", "Tocar.mp4", "Todo.mp4", "Tomar.mp4", "Tomate.mp4",
    "Torta.mp4", "Tortuga.mp4", "Trauma.mp4", "Treinta.mp4", "Turismo.mp4", "Turrialba.mp4", "Turrucares.mp4", "TÉ.mp4", "TÉ^FRIO.mp4", "TÍO-2.mp4",
    "TÍO.mp4", "ULTIMO.mp4", "UNICO.mp4", "UNIVERSIDAD.mp4", "USTED.mp4", "Uno-Mitad.mp4", "Uruguay.mp4", "Usar.mp4", "VACA.mp4", "VACÍO.mp4",
    "VEINTE.mp4", "VER-3.mp4", "VERDAD.mp4", "VIOLAR-LA-LEY.mp4", "VOLAR-EN-AVIÓN.mp4", "VOLCÁN.mp4", "Vacaciones.mp4", "Vapor.mp4", "Vegetal.mp4", "Vencer.mp4",
    "Vender.mp4", "Ver_6.mp4", "Verano.mp4", "Viejo.mp4", "Viento.mp4", "Viernes.mp4", "Vocabulario.mp4", "WINDOWS.mp4", "WWW.mp4", "YA-2.mp4",
    "YA.mp4", "YO.mp4", "Yuca.mp4", "ZARCERO-2.mp4", "ZARCERO.mp4", "Zacate.mp4", "Zapato.mp4", "Zapote.mp4", "como 2.mp4", "taxi.mp4"
    ])
    #30 videos of data
    number_sequences = 30
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
