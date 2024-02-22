FROM nvidia/cuda:12.1.0-base-ubuntu22.04
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 && \
    apt-get install -y python3-pip
# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

RUN mv pyproject.toml.docker pyproject.toml

# Install poetry
RUN pip install poetry

# Use poetry to install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# setting PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# setting environment variables

# Make port 8010 available to the world outside this container
EXPOSE 8031

# Run the command to start uWSGI
CMD uvicorn api:app --host 0.0.0.0 --port 8031