FROM nginx:alpine

# Copy nginx configuration files
COPY default.conf /etc/nginx/templates/default.conf.template

# Copy static files
COPY ./public /usr/share/nginx/html

# Set default value for the environment variable
# This allows for proper service discovery in different environments
# In Coolify, use the provided service name or FQDN if available
ENV WHISPER_API_HOST=${COOLIFY_SERVICE_NAME:-${COOLIFY_CONTAINER_NAME:-whisper-api}}

# Nginx will automatically use the template files in /etc/nginx/templates/
# and substitute environment variables when the container starts

# Expose port 80
EXPOSE 80 