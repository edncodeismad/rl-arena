FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /usr/src/app
COPY snake_cnn.py . 
RUN pip install --no-cache-dir torchrl tensordict
CMD ["python", "snake_cnn.py"]