After docker-compose has been successfully launched

Run Synthea load and transform script

## To build

docker build --no-cache -t onc.sml.synthea .

## To run small batch
docker run --network="host"  -v /$(pwd)/fhir:/oncfhirinput -it  synthea


## To run larger batch
docker run --network="host"  -v /$(pwd)/fhir2:/oncfhirinput -it  synthea
