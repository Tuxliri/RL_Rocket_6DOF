FROM python:3.7

RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*

COPY my_environment /my_environment
COPY config.yaml .
COPY main_6DOF.py .
COPY setup.py .
COPY requirements.txt .
COPY docker_startup.sh .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Make the start script executable
RUN ["chmod", "+x", "./docker_startup.sh"]

# The API key MUST be passed as an environmental variable to the container
ENTRYPOINT [ "./docker_startup.sh" ]
