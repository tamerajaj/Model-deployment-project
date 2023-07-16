# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the application code to the working directory
COPY . /app


# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry install --no-dev

# Expose the port
EXPOSE 9696

# Run the application
CMD ["poetry", "run","uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]