# Use an official lightweight Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy Streamlit UI script
COPY app.py .

# Install required dependencies
RUN pip install --no-cache-dir streamlit requests

# Expose the Streamlit default port
EXPOSE 8501

# Start Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
