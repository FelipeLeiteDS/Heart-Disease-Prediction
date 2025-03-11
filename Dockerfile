# Python Version
FROM python:3.12-slim

# Define directory
WORKDIR /Heart-Disease-Prediction

# Installing libraries
RUN pip install pandas numpy sklearn matplotlib.pyplot seaborn

# Copy .py file
COPY Random-Forest_Confusion-Matrix.py .

# Execute command
CMD ["python", "Random-Forest_Confusion-Matrix.py"]
