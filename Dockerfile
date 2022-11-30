FROM python:3.7

RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*

COPY my_environment /my_environment
COPY config.yaml .
COPY main_6DOF.py .
COPY setup.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .
RUN wandb login 0cbac35bfe601a8c17f4132f2fb22bb9a9b03e40

CMD ["python", "main_6DOF.py"]