# Use a specific version of the Python image
FROM python:3.9.17-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=true
ENV APP_HOME=/back-end
WORKDIR $APP_HOME

# Copy application code to the container
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
