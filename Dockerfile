# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos y el resto de la aplicación al contenedor
COPY requirements.txt /app/

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . /app/

# Exponer el puerto en el que FastAPI va a correr
EXPOSE 80

# Definir el comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
