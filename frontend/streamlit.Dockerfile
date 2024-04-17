FROM python:3.10-slim

WORKDIR /app

# Copy the directory contents into the container at /app
COPY . .

# Install streamlit in the python container
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]