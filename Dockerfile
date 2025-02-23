# Use the official PyTorch image as the base
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

#install CURL
RUN apt-get update && apt-get install -y curl && apt-get clean

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade sentence-transformers huggingface_hub
RUN pip install -U langchain-community
RUN pip install --upgrade langchain 
RUN pip install langchain-experimental
RUN pip install -qU langchain-community langchain-elasticsearch
RUN pip install tabulate
RUN pip install langchain-openai




# Copy application code
COPY . .

# Expose port
EXPOSE 8000


# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

