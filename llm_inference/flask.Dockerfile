# Start from NVIDIA CUDA base image to enable GPU access in container
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
CMD nvidia-smi # To check if CUDA is accessible

# working directory in container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Flask
RUN pip3 install Flask

# Copy requirements file into the container
COPY requirements.txt .

# Copy the rest of the application code into the container
COPY . .

# Install required packages
RUN pip install -r requirements.txt
#RUN pip install -i https://pypi.org/simple/ bitsandbytes
#RUN pip install flash-attn==2.5.7 --no-build-isolation

# Expose the port that Flask will run on
EXPOSE 5000

#Define ENV variables
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=True

# Set the command to run the LLM inference endpoint for topic generations
CMD ["python3", "inference_topic_gen.py"]