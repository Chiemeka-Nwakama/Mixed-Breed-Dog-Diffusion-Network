FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code

COPY . .

# Copies the weights into image for docker container when running
COPY checkpoints/conditional/stage2_conditional.pth /checkpoints/conditional/stage2_conditional.pth

ENV PYTHONUNBUFFERED=1

#path of config file with model presets
ENV CONFIG_PATH=/app/config.yml

#path to load model weights
ENV CKPT_PATH=/checkpoints/conditional/stage2_conditional.pth

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
