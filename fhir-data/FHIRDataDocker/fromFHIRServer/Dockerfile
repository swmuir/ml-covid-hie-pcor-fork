FROM python:3.8
RUN pip install requests
COPY fromfhirserver.py /tmp/
RUN mkdir /oncfhir
CMD ["python", "/tmp/fromfhirserver.py"]