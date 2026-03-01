FROM python:3.12
WORKDIR /app
COPY . . 
COPY requirements/requirements.txt /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/requirements.txt \
  && pip install --no-cache-dir jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
