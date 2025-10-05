# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file first to take advantage of Docker's layer caching
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Download NLTK data needed for feature extraction
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy the app and data folders into the container
COPY ./app /code/app
COPY ./data /code/data

# Run the training script to build the model files INSIDE the container
# This makes our deployment self-contained and repeatable
RUN python -m app.train

# Tell the world that the container is listening on port 8080
EXPOSE 8080

# The command to run when the container starts.
# We use 0.0.0.0 to make it accessible from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]