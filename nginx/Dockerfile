FROM nginx:alpine

# Copy nginx configuration files
COPY default.conf /etc/nginx/templates/default.conf.template

# Copy static files
COPY ./html /usr/share/nginx/html

# Set default value for the environment variable
# This will be overridden by the COOLIFY_CONTAINER_NAME in Coolify environments
ENV WHISPER_API_HOST=${COOLIFY_CONTAINER_NAME:-whisper-api}

# Nginx will automatically use the template files in /etc/nginx/templates/
# and substitute environment variables when the container starts

# Expose port 80
EXPOSE 80 