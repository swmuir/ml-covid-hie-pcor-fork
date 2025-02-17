cd generate
 
docker run --network="host"  -v "%cd%/output:/oncmloutput"    -e RUNQA='True'  -it  onc.sml.generate
 
docker run --network="host"  -v "%cd%//output:/oncmloutput"  -it   onc.sml.embeddings
 
cd ..

