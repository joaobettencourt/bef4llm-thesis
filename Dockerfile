# Base image with Python 3.13
FROM python:3.13

# Set working directory inside container
WORKDIR /app

# Install system dependencies for cvxopt, pm4py, Java (for Tika), and others
RUN apt-get update && apt-get install -y \
    build-essential \
    glpk-utils \
    libglpk-dev \
    liblapack-dev \
    libopenblas-dev \
    git \
    curl \
    default-jdk \
 && rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy Python requirements
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the project code
COPY src/ ./src
COPY setup.py .
COPY README.md .

# Set PYTHONPATH so Python can find your modules
ENV PYTHONPATH=/app/src

# Default command when container starts
CMD ["python3"]