cd org.onc.ml.transformation

docker build -t onc.sml.transformation:latest .
 
 cd ..

cd onc-sml-sandbox

docker build -t onc.sml.fhir:latest .
 
cd ..

docker-compose -f docker-compose-synthea.yml up  --wait

Start-Sleep -Seconds 60

cd synthea

docker build --no-cache -t onc.sml.synthea .

docker run --network="host"  -v C://Users//esrav//Downloads//oncml-phase2//FHIRDataDocker//synthea//fhir:/oncfhirinput -it  onc.sml.synthea  
#docker run --network="host"  -v /$(pwd)/fhir:/oncfhirinput -it  onc.sml.synthea 

cd ..

cd generate

docker build -f DockerfileGenerate -t onc.sml.generate .
 
docker build -f DockerfileEmbeddings -t onc.sml.embeddings . 

docker run --network="host"  -v C://Users//esrav//Downloads//oncml-phase2//FHIRDataDocker//generate//output:/oncmloutput  -it  onc.sml.generate

docker run --network="host"  -v C://Users//esrav//Downloads//oncml-phase2//FHIRDataDocker//generate//output:/oncmloutput  -it   onc.sml.embeddings
#docker run --network="host"  -v $(pwd)/output:/oncmloutput  -it  onc.sml.generate

#docker run --network="host"  -v $(pwd)/output:/oncmloutput  -it   onc.sml.embeddings

cd ..

docker-compose -f docker-compose-synthea.yml down