FROM python:3.7

RUN apt-get update && apt-get install -y xvfb libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY my_environment /my_environment/my_environment
COPY setup.py /my_environment
COPY main_6DOF_ray.py .
COPY docker_startup.sh .

RUN pip install -e /my_environment

# Make the start script executable
RUN ["chmod", "+x", "./docker_startup.sh"]

# The API key MUST be passed as an environmental variable to the container
ENTRYPOINT [ "./docker_startup.sh" ]
