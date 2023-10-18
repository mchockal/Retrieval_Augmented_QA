# Use the base Weaviate image
FROM semitechnologies/weaviate:1.21.7

# Expose port 8080 for the application
EXPOSE 8080

# Define a volume for data persistence
VOLUME /var/lib/weaviate

# Set environment variables
ENV QUERY_DEFAULTS_LIMIT=25
ENV AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
ENV PERSISTENCE_DATA_PATH=/var/lib/weaviate
ENV DEFAULT_VECTORIZER_MODULE=none
ENV CLUSTER_HOSTNAME=node1

# Set the restart policy
CMD ["weaviate", "--restart", "on-failure:0"]