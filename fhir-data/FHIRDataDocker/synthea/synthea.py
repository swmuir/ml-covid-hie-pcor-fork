# import OS module
import os
import requests

 

isDebug = False
loadSynthea = True

# Get the list of all files and directories
fhirInputPath = "/oncfhirinput"

# to store files in a list
list = []

fhir_url = "http://localhost:8080/fhir"


if loadSynthea:
    print("Start load")    
    for (root, dirs, file) in os.walk(fhirInputPath):
        for f in file:
            if 'hospitalInformation' in f:
                print('Load Hospital ' ,f)
                print(os.path.join(root, f))
                hospitalFile = open(os.path.join(root, f),"rb")
                headers = {'Content-type': 'application/json'}
                test_response = requests.post(fhir_url, data=hospitalFile, headers=headers)
                if test_response.ok:
                    print("Upload completed successfully!")
                else:
                    print("Something went wrong!")  
                    print(test_response.text)          


    for (root, dirs, file) in os.walk(fhirInputPath):
        for f in file:
            if 'practitionerInformation' in f:
                print('Load Practitioner ',f)
                practitionerFile = open(os.path.join(root, f),"rb")
                headers = {'Content-type': 'application/json'}
                test_response = requests.post(fhir_url, data=practitionerFile, headers=headers)
                if test_response.ok:
                    print("Upload completed successfully!")
                else:
                    print("Something went wrong!")  
                    print(test_response.text)               
    patientCounter = 0
    for (root, dirs, file) in os.walk(fhirInputPath):
        for f in file:
            if not 'practitionerInformation' in f and not 'hospitalInformation' in f:
                print('Load Patient ',f)
                patientFile = open(os.path.join(root, f),"rb")
                headers = {'Content-type': 'application/json'}
                test_response = requests.post(fhir_url, data=patientFile, headers=headers)
                if test_response.ok:
                    print("Upload completed successfully!")
                    patientCounter += 1
                else:
                    print("Something went wrong!")  
                    print(test_response.text)   
    print("Pause for processing")
