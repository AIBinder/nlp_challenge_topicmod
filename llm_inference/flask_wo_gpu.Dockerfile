FROM python:3.10-slim

# Set environment variables for Flask
ENV FLASK_APP=inference_topic_gen.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install Git (probably required for flash attention install)
#RUN apt-get update && apt-get install git -y

# Copy the requirements file into the container
COPY requirements.txt .

# Install required packages
RUN pip install -r requirements.txt
RUN pip install -i https://pypi.org/simple/ bitsandbytes
#RUN pip install flash-attn==2.5.7 --no-build-isolation

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Flask will run on
EXPOSE 5000

# Run the Flask application
CMD ["flask", "run"]