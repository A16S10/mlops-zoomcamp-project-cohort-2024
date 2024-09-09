FROM python:3.11.7-slim

# Set working directory
WORKDIR /opt/ml/code

# Copy the model and app.py file
COPY app.py /opt/ml/code/app.py
COPY models/lin_reg.bin /opt/ml/model/lin_reg.bin


# Install necessary libraries
RUN pip install pandas scikit-learn gradio flask numpy pyarrow s3fs 

# Expose the Gradio and Flask ports
EXPOSE 8080 8081

# Run the Gradio app (this includes Flask in a separate thread)

ENTRYPOINT ["python", "/opt/ml/code/app.py"]
CMD ["serve"] 
