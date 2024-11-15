This is a set of python scripts used to generate the initial csv/model file 
The dependencies are the  fhir endpoints have been started as well as the transformation

## generate.py
Will call 3 sets of cqlplans operations
bundleofbundle per encounter
bundleofbundle per patient
bundleofbundle qa plan

## embeddings.py
This will process the notes from the generate step and apply embedding logic to de-indentify the notes columns

## To build

docker build -f DockerfileGenerate -t onc.sml.generate .
 
docker build -f DockerfileEmbeddings -t onc.sml.embeddings . 

## To run

Run all patients in cache and qa process

docker run --network="host"  -v /$(pwd)/output:/oncmloutput    -e RUNQA='True'  -it  onc.sml.generate

Limit the number of patients

docker run --network="host"  -v /$(pwd)/output:/oncmloutput    -e RUNQA='False' -e MAXPATIENTS='10' -it  onc.sml.generate


docker run --network="host"  -v /$(pwd)/output:/oncmloutput  -it   onc.sml.embeddings

go.scr
go.scr will build an launch the scripts