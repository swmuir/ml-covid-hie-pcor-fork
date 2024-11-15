# import OS module
import os
import requests
import time
import json
import csv

 

isDebug = False 
loadSynthea = True

# Get the list of all files and directories
mlOutputPath = "/oncmloutput"

# to store files in a list
list = []

# fhir_url = "http://localhost:8181/fhir"
fhir_url = USER = os.getenv('FHIRSERVER')

mdmi_url = "http://localhost:8282/mdmi/transformation"

# if loadSynthea:
#     print("Start load")    
#     for (root, dirs, file) in os.walk(fhirInputPath):
#         for f in file:
#             if 'hospitalInformation' in f:
#                 print('Load Hospital ' ,f)
#                 print(os.path.join(root, f))
#                 # print(os.path.realpath(f))
#                 hospitalFile = open(os.path.join(root, f),"rb")
#                 headers = {'Content-type': 'application/json'}
#                 test_response = requests.post(fhir_url, data=hospitalFile, headers=headers)
#                 # test_response = requests.post(fhir_url, files = {"form_field_name": hospitalFile},headers)
#                 if test_response.ok:
#                     print("Upload completed successfully!")
#                     # print(test_response.text)
#                 else:
#                     print("Something went wrong!")  
#                     print(test_response.text)          


#     for (root, dirs, file) in os.walk(fhirInputPath):
#         for f in file:
#             if 'practitionerInformation' in f:
#                 print('Load Practitioner ',f)
#                 practitionerFile = open(os.path.join(root, f),"rb")
#                 headers = {'Content-type': 'application/json'}
#                 test_response = requests.post(fhir_url, data=practitionerFile, headers=headers)
#                 # test_response = requests.post(fhir_url, files = {"form_field_name": hospitalFile},headers)
#                 if test_response.ok:
#                     print("Upload completed successfully!")
#                     # print(test_response.text)
#                 else:
#                     print("Something went wrong!")  
#                     print(test_response.text)               
#     patientCounter = 0
#     for (root, dirs, file) in os.walk(fhirInputPath):
#         for f in file:
#             if not 'practitionerInformation' in f and not 'hospitalInformation' in f:
#                 print('Load Patient ',f)
#                 patientFile = open(os.path.join(root, f),"rb")
#                 headers = {'Content-type': 'application/json'}
#                 test_response = requests.post(fhir_url, data=patientFile, headers=headers)
#                 # test_response = requests.post(fhir_url, files = {"form_field_name": hospitalFile},headers)
#                 if test_response.ok:
#                     print("Upload completed successfully!")
#                     patientCounter += 1
#                     # print(test_response.text)
#                 else:
#                     print("Something went wrong!")  
#                     print(test_response.text)   
#     print("Pause for processing")
#     time.sleep(3)


xpheaders = {'Content-type': 'application/json'}
xbofb_response = requests.get(fhir_url+'/Patient',  headers=xpheaders) 
# print(xbofb_response.text)

destination_file = open(mlOutputPath+'/mloutput.csv', 'w', newline='')
csv_writer = csv.writer(destination_file)

isFirst = True
json_object = json.loads(xbofb_response.text)
for _entry in json_object["entry"]: 
    print(_entry["resource"]["id"])
    pheaders = {'Content-type': 'application/xml'}
    print(fhir_url+'/$bundleofbundles?patient='+_entry["resource"]["id"])
    bofb_response = requests.get(fhir_url+'/$bundleofbundles?patient='+_entry["resource"]["id"],  headers=pheaders)     
    if bofb_response.ok:
        print("bundleofbundles successfully!")
        if isDebug:
            with open(mlOutputPath+"/bofb"+ str(_entry["resource"]["id"]) + ".xml", "w+") as text_file:
                text_file.write(bofb_response.text)

    
        headers = {
                'Content-Type': 'application/xml'
        }
        transformationResult = requests.request("POST", "http://localhost:8282/mdmi/transformation/byvalue/?source=FHIRR4JSON.MasterBundle&target=ONCML.DIABETES", headers=headers, data=bofb_response.content)

    

        with open(mlOutputPath+"/mlmodel"+ str(_entry["resource"]["id"]) + ".csv", "w+") as text_file:
            text_file.write(str(transformationResult.content, encoding='utf-8'))

        file1 = open(mlOutputPath+"/mlmodel"+ str(_entry["resource"]["id"]) + ".csv", 'r')
        csv_reader1 = csv.reader(file1)

        print(isFirst)  
        if isFirst:
            headers = next(csv_reader1)
            csv_writer.writerow(headers)
            isFirst = False
            print(isFirst)
        else:
            next(csv_reader1, None)  # skip the headers    
        # print(isFirst)      

        for row in csv_reader1:
            csv_writer.writerow(row)

        if not isDebug:
            try:
                os.remove(mlOutputPath+"/mlmodel"+ str(_entry["resource"]["id"]) + ".csv")
            except OSError:
                pass    

    else:
        print("Something went wrong!")  
        print(bofb_response.text)                 
    # time.sleep(1)

destination_file.close()    
    
 