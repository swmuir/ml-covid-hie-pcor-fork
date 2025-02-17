cd synthea
 
docker run --network="host"  -v "%cd%/fhir:/oncfhirinput" -it  onc.sml.synthea
 
cd ..

