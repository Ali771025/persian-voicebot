FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential cmake pkg-config \
    libsndfile1 espeak-ng \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp /opt/whisper.cpp \
  && cmake -S /opt/whisper.cpp -B /opt/whisper.cpp/build -DWHISPER_FFMPEG=OFF \
  && cmake --build /opt/whisper.cpp/build -j
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app /app/app
COPY web /app/web
COPY scripts /app/scripts
COPY README.md /app/README.md
ENV WHISPER_CLI=/opt/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_MODEL=/models/whisper/ggml-small.bin
ENV MEDIA_DIR=/data/media
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
