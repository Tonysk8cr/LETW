FROM python:3.8-slim

WORKDIR /LETW

# Instalar librerías del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements primero
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Comando por defecto
CMD ["python", "Model/Test/App.py"]
