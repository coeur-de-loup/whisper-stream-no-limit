FROM nginx:alpine

# Copy nginx configuration files
COPY default.conf /etc/nginx/templates/default.conf.template

# Copy static files
COPY ./public /usr/share/nginx/html

# Set default value for the environment variable with the specific container name
# This is hardcoded to the exact container name in Coolify
ENV WHISPER_API_HOST=whisper-api-m0o808wc4gwkk8kggg4gowcg-215423103736

# Nginx will automatically use the template files in /etc/nginx/templates/
# and substitute environment variables when the container starts

# Expose port 80
EXPOSE 80 