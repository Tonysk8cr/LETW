# LETW

LETW es una iniciativa de código abierto dedicada al desarrollo de un modelo de IA capaz de reconocer y comprender signos de LESCO (Lengua de Señas Costarricense). Este proyecto proporciona herramientas y guías para construir un modelo basado en TensorFlow que facilite la interpretación de la lengua de señas y apoye la comunicación inclusiva.

Para Inglés (For English): [![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/Tonysk8cr/LETW/blob/Dev/README.md)


## Instalación

> [!IMPORTANT]
> Usa el código de la rama `main`. La rama `dev` es para desarrollo y puede contener errores.

Primero, clona el repositorio:
```bash
git clone https://github.com/Tonysk8cr/LETW.git
cd LETW
```

Existen dos formas de configurar el entorno del proyecto. Recomendamos utilizar la herramienta `uv`.

### Configuración recomendada con `uv`

Este proyecto utiliza `uv` para una gestión de dependencias rápida y fiable. Utilizará automáticamente la versión correcta de Python definida para el proyecto.

1.  **Instala `uv`**

    Si no tienes `uv` instalado, sigue las instrucciones de instalación en la [página oficial del proyecto uv](https://astral.sh/uv/install).

2.  **Crea el entorno virtual y sincroniza las dependencias**

    `uv` creará el entorno virtual e instalará todas las dependencias del archivo lock (`uv.lock`) en un solo paso.
    ```bash
    uv sync
    ```

3.  **Activa el entorno virtual**
    ```bash
    # Windows:
    .venv\Scripts\activate

    # Linux o MacOs:
    source .venv/bin/activate
    ```
    Ya puedes pasar a la sección [Uso / Ejemplos](#uso--ejemplos). Para ejecutar la aplicación principal, puedes usar `uv run ./Model/Test/App.py`.

### Configuración alternativa con `pip`

Este método está disponible si prefieres gestionar tu entorno manualmente. Sin embargo, no es el método principal soportado.

1.  **Instala Python**

    Asegúrate de tener Python 3.8 instalado en tu sistema.

2.  **Crea y activa el entorno virtual**
    ```bash
    python -m venv .venv
    ```
    ```bash
    # Windows:
    .venv\Scripts\activate

    # Linux o MacOs:
    source .venv/bin/activate
    ```

3.  **Instala las dependencias**

    Esto instalará los paquetes listados en `requirements.txt`. Ten en cuenta que este archivo puede no estar tan actualizado como el archivo `uv.lock`.
    ```bash
    pip install -r requirements.txt
    ```


## Uso / Ejemplos

Una vez que hayas instalado todas las dependencias necesarias, podrás ejecutar el proyecto.
Es importante notar que podrían requerirse algunos cambios en el código; actualmente, el código está configurado para minimizar esto, pero sigue siendo importante revisarlo.

Este sistema se accede a través del archivo App.py, que sirve como punto de entrada principal. Administra el flujo de la aplicación y facilita la ejecución del sistema.

1. Crea las carpetas necesarias
Al ejecutar App.py (ya que es una aplicación de consola), verás salida en la consola. En el menú principal, la primera opción que debes elegir es la opción 1. Esto creará dos carpetas críticas dentro de la carpeta /Test de tu directorio de Modelo:

MP_Data: Aquí se almacenarán posteriormente los arrays de NumPy.

Test_Videos: Aquí almacenarás tus videos de entrenamiento.

Ten en cuenta que, una vez creada automáticamente la carpeta Test_Videos, también se generará una subcarpeta para cada acción especificada en la clase principal bajo la variable llamada signs. Por favor, asegúrate de que la lista de acciones esté correctamente definida en esa clase.

Después, deberás colocar manualmente los videos correspondientes en cada una de estas carpetas de acción. Los videos deben seguir la convención de nombres:
Action(1, 2, 3, 4) (por ejemplo: Hello1.mp4, Hello2.mp4, etc.).


2. Verifica rutas de archivos incorrectas

Aunque el proyecto utiliza rutas relativas en su lógica, es importante revisarlas. Como este proyecto fue desarrollado y probado originalmente en un entorno local, debes asegurarte de que todas las rutas de archivos estén correctamente configuradas para tu sistema.

Revisa los siguientes archivos y actualiza cualquier ruta codificada:

App.py

DataExtraction.py

DataLabelling.py

RealtimePrediction.py

Asegúrate de que todas las rutas apunten a las ubicaciones correctas en tu máquina para evitar problemas durante la ejecución.

Si estás usando una distribución Linux, puedes editar los archivos usando:
```bash
vim <path/class.py>
```


3. App.py
Como se mencionó anteriormente, App.py es el punto de entrada principal del proyecto.
Al ejecutarlo, aparecerá un menú en la consola desde el cual puedes navegar y ejecutar las diferentes funcionalidades del sistema.




## Contribuciones
Como se mencionó al inicio de este README, este proyecto es open source.
Siéntanse libre de usarlo y, si lo hace, trata de no cobrar a otros por él.
Recuerda que este proyecto fue desarrollado principalmente para ayudar a quienes más lo necesitan.


## Autores

- [@Tonysk8cr](https://github.com/Tonysk8cr)

## Créditos

Este proyecto esta basado en [ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage)
by [@nicknochnack](https://github.com/nicknochnack)
Partes de su código fueron adaptados a las necesidades de nuestro proyecto