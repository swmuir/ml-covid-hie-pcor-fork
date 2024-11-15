FROM python:3.8
RUN pip install requests
COPY synthea.py /tmp/
RUN mkdir /oncfhir
CMD ["python", "/tmp/synthea.py"]