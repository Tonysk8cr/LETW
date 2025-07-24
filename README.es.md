# LETW

LETW es una iniciativa de código abierto dedicada al desarrollo de un modelo de inteligencia artificial que reconoce y comprende señas de LESCO (Lengua de Señas Costarricense). Este proyecto brinda herramientas y orientación para construir un modelo basado en TensorFlow que facilite la interpretación de la lengua de señas y promueva una comunicación más inclusiva.

For English: [![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/Tonysk8cr/LETW/blob/Dev/README.md)


## Instalación

¡Utiliza el código disponible en la rama principal (main), ya que la rama de desarrollo (Dev) está en proceso y podría contener errores!
Este proyecto requiere Python 3.8 debido al uso de Mediapipe Holistic.

1. Clona el repositorio
```bash
git clone https://github.com/Tonysk8cr/LETW.git
```
2. Crea el entorno virtual
```bash
python -m venv .venv
```
3. Activa el entorno virtual
```bash
# Windows:
.venv\Scripts\activate

# Linux o MacOs:
source .venv/bin/activate
```
4. Instala las dependencias
```bash
pip install -r requirements.txt
```


## Uso / Ejemplos

Una vez que hayas instalado todas las dependencias necesarias, deberías poder ejecutar el proyecto sin problemas.
Es importante tener en cuenta que podrían requerirse algunos ajustes en el código. Actualmente, el código está configurado para evitar esto, pero aun así es recomendable revisarlo.

Este sistema se accede a través del archivo App.py, que funciona como el punto de entrada principal. Administra el flujo de la aplicación y facilita su ejecución.

1. Crea las carpetas necesarias
Cuando ejecutes el archivo App.py (ya que se trata de una aplicación de consola), verás algunas salidas en la consola. En el menú principal, selecciona la opción 1. Esto creará dos carpetas clave dentro de la carpeta /Test ubicada en el directorio Model:

MP_Data: Aquí se almacenarán posteriormente los arreglos de NumPy.

Test_Videos: Aquí deberás colocar tus videos de entrenamiento.

Ten en cuenta que una vez que se cree automáticamente la carpeta Test_Videos, también se generará una subcarpeta por cada acción definida en la clase DataExtraction. Por lo tanto, asegúrate de que la lista de acciones esté correctamente definida en dicha clase.

Después de esto, deberás colocar manualmente los videos correspondientes dentro de cada carpeta de acción. Los videos deben seguir el siguiente formato de nombre:
Acción(1, 2, 3, 4) (por ejemplo: Hola1.mp4, Hola2.mp4, etc.).


2. Verifica rutas de archivos incorrectas

Como este proyecto fue desarrollado y probado inicialmente en un entorno local, es necesario que verifiques que todas las rutas estén correctamente configuradas para tu sistema.

Revisa los siguientes archivos y actualiza cualquier ruta escrita de forma fija (hardcoded):

App.py

DataExtraction.py

DataLabelling.py

RealtimePrediction.py

Asegúrate de que todas las rutas apunten a las ubicaciones correctas en tu máquina para evitar errores durante la ejecución.


3. App.py
Como se mencionó anteriormente, App.py es el punto de entrada principal del proyecto.
Una vez ejecutado, aparecerá un menú en la consola desde el cual podrás navegar y utilizar las distintas funcionalidades del sistema.




## Contribuciones
Como se mencionó al inicio de este README, este es un proyecto de código abierto.
Siéntete libre de utilizarlo, y si lo haces, trata de no cobrar por él.
Ten en cuenta que este proyecto fue desarrollado principalmente para ayudar a quienes más lo necesitan.


## Autores

- [@Tonysk8cr](https://github.com/Tonysk8cr)

