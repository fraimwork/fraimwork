# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set the entrypoint command to run the Flask app
CMD ["python", "app.py"]