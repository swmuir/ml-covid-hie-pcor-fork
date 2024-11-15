## FHIRDataDocker Data Stream
The set of files and projects in the FHIRDataDocker are designed to 
* Load synthea data into intermediate FHIR Server (HAPI)
* Apply CQL for information enrichment requirements
* Extract FHIR content in Encounter Bundles
* Transform/Flatten Encounter Bundles into ML input file

## Data Stream Components
### onc-sml-sandbox
Local FHIR Cache of cohort
### org.onc.ml.transformation
Transformation services which converts the FHIR bundles content to ML 
### synthea
Upload synthea data for testing cohort
### generate
Transforms data from Local FHIR Cache into ML 
#### onc.sml.generate
Pulls down FHIR bundles and creates csv files

#### onc.sml.embeddings
Applies embeddings steps for encounter notes

## Build All Docker Images Step
execute ./build.scr

## Run Options
### Load Synthea1000 and Run
This will load 1k patients and build the ML files
execute ./loadandrunsynthea1000.scr

###  Run after Load
This will build the ML files from the current set of patients in the FHIR Cache
execute ./runafterload.scr



 ## Manual Steps to run the data stream
### Step 1 ONC Sandbox 
cd ./onc-sml-sandbox 
docker build -t onc.sml.fhir:latest .
docker-compose up -d
cd ..

### Step 2 Transformation Service 
cd org.onc.ml.transformation 
mvn clean install 
docker build -t onc.sml.transformation:latest .
docker-compose up -d
cd ..
### Step 3 
cd synthea
docker build --no-cache -t onc/onc.sml.synthea .
docker run --network="host"  -v /$(pwd)/fhir:/oncfhirinput -it  onc.sml.synthea 
cd .. 


### Step 3 
cd generate
docker build -f DockerfileGenerate -t onc.sml.generate .
 
docker build -f DockerfileEmbeddings -t onc.sml.embeddings . 

docker run --network="host"  -v /$(pwd)/output:/oncmloutput  -it  onc.sml.generate

docker run --network="host"  -v /$(pwd)/output:/oncmloutput  -it   onc.sml.embeddings



