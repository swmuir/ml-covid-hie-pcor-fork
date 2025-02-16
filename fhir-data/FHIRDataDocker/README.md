cd synthea
docker build --no-cache -t onc.sml.synthea .
cd ..

cd gov-onc-sml-fhir 
docker build -t onc.sml.fhir:latest .
cd ..

cd generate
docker build -f DockerfileGenerate -t onc.sml.generate .
docker build -f DockerfileEmbeddings -t onc.sml.embeddings .
cd ..
