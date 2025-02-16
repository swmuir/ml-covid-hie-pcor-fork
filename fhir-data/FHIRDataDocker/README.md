# Build Instructions

cd synthea
docker build --no-cache -t onc.sml.synthea .
cd ..

cd gov-onc-sml-fhir 
docker build -t onc.sml.fhir:latest .
cd ..

cd org.onc.ml.transformation
docker build -t onc.sml.transformation:latest .
cd ..

cd generate
docker build -f DockerfileGenerate -t onc.sml.generate .
docker build -f DockerfileEmbeddings -t onc.sml.embeddings .
cd ..

# Load Instructions
cd synthea

docker run --network="host"  -v /$(pwd)/fhir:/oncfhirinput -it  onc.sml.synthea
cd..

# Run Instructions
cd generate

docker run --network="host"  -v /$(pwd)/output:/oncmloutput    -e RUNQA='True'  -it  onc.sml.generate

docker run --network="host"  -v /$(pwd)/output:/oncmloutput  -it   onc.sml.embeddings

cd ..