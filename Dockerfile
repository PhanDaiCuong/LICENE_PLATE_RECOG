FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements (tự tạo file requirements.txt)
COPY requirements.txt /app/requirements.txt

# Cài python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code vào /app
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]
