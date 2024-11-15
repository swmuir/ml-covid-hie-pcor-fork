import concurrent.futures
import logging
# import threading
import time
import requests
import time
import json
import csv
from itertools import repeat
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def getPatientIds(maxpatients):
    isDebug = True
    patientList = []
    logging.info("============== Gathering Patients ==============")
    xpheaders = {'Content-type': 'application/json'}
    xbofb_response = requests.get(fhir_url+'/Patient',  headers=xpheaders)
    hasNext = True
    json_object = json.loads(xbofb_response.text)
    ctr = 0
    while hasNext:
        for _entry in json_object["entry"]:
            patientId = _entry["resource"]["id"]
            logging.info(f"Captured Patient ID: {patientId}")
            patientList.append(_entry["resource"]["id"])
            ctr += 1
            if ctr > maxpatients :
                return patientList
        nextLink = None
        logging.info(f"Response Keys: {json_object.keys()}")
        if 'link' in json_object:
            logging.info(f"Link in response {json_object['link']}")
            for link in json_object['link']:
                if link['relation'] == 'next':
                    nextLink = link['url']
                    break
        else:
            logging.info(f"Link not found in response...")

        if nextLink:
            logging.info(f"Next Link Found: {nextLink}")
            xbofb_response = requests.get(nextLink, headers=xpheaders)
            json_object = json.loads(xbofb_response.text)
        else:
            logging.info("Next Link is MISSING...")
            hasNext = False
    logging.info("============== Finished Retrieving Patients ==============")
    return patientList

def combine_data(thePlan, theContext, thePatients, writer):
    isFirst = True
    index_map = {}
    index_count = 1
    for patient in thePatients:
        index_map[patient] = index_count
        context_file = f"/oncmloutput/{theContext}{thePlan}{patient}.csv"
        print(f"Starting to parse results for (Patient | Plan | Context): {patient} | {thePlan} | {theContext}")
#        print(f"LOADING FILE: Starting to parse results for (Patient | Plan | Context): {patient} | {thePlan} | {theContext}")
        file1 = open(context_file, 'r')
        print(f"Parsing {context_file}")
        csv_reader1 = csv.reader(file1)
        print(f"First row of file: {isFirst}")
        if isFirst:
            headers = ['Patient IDX'] + next(csv_reader1)
            writer.writerow(headers)
            isFirst = False
        else:
            next(csv_reader1, None)  # skip the headers         
        for row in csv_reader1:
            writer.writerow([ index_count ] + row)
        index_count += 1
        file1.close()
    with open(f"/oncmloutput/{theContext}_{thePlan}_PATIENT_ID_MAPPINGS.csv","w") as w:
        w.write('Patient ID,Patient IDX\n')
        for patid, idx in index_map.items():
            w.write(f"{patid},{idx}\n")
            

def runstats():
    start_time = time.time()

    # Create the stats directory if it doesn't exist
    stats_dir = '/oncmloutput/stats'
    os.makedirs(stats_dir, exist_ok=True)
    
    # Timing
    def log_time(message):
        print(f"{message} took {time.time() - start_time:.2f} seconds")
    
    print("Loading DataFrame...")
    df = pd.read_csv('/oncmloutput/' + 'cqlplan' + 'file.csv')
    log_time("Loading DataFrame")
    
    print("Calculating summary statistics...")
    summary_statistics = df.describe(include='all')
    log_time("Calculating summary statistics")
    
    total_rows = df.shape[0]

    print("Calculating null values...")
    null_counts = df.isnull().sum()
    log_time("Calculating null values")
    
    non_null_counts = summary_statistics.loc['count'] if 'count' in summary_statistics.index else df.count()

    print("Calculating unique value counts...")
    unique_counts = df.nunique()
    log_time("Calculating unique value counts")
    
    yVar = [
        "Chronic kidney disease all stages (1 through 5)",
        "Acute Myocardial Infarction",
        "Hypertension Pulmonary hypertension",
        "Ischemic Heart Disease",
    ]
    yVarList = {
        "Diabetes": [
            "Type 1 Diabetes",
            "Type II Diabetes",
        ]
    }

    print("Calculating class imbalance and label co-occurrence matrix...")
    label_columns = yVar + [label for sublist in yVarList.values() for label in sublist]
    class_imbalance = df[label_columns].sum().sort_values(ascending=False) if label_columns else None
    label_cooccurrence_matrix = df[label_columns].T.dot(df[label_columns])
    log_time("Calculating class imbalance and label co-occurrence matrix")

    # List of columns to remove
    columns_to_remove = [
    'Patient IDX', 'SNOMED Codes 0', 'SNOMED Codes 1', 'SNOMED Codes 2', 
    'SNOMED Codes 3', 'ICD-10 Codes 0', 'ICD-10 Codes 1', 'ICD-10 Codes 2', 
    'ICD-10 Codes 3', 'ICD-10 Codes Other', 'Smoking Status', 
    'Procedure Codes 0', 'Procedure Codes 1', 'Procedure Codes 2', 'Procedure Codes 3'
    ]

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    nonnumeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # Remove the specified columns
    numeric_columns = numeric_columns.drop(columns_to_remove, errors='ignore')

    #print(numeric_columns)
    
    print("Calculating correlation matrix...")
    correlation_matrix = df[numeric_columns].corr()
    log_time("Calculating correlation matrix")

    print("Saving statistics to CSV...")
    with open(os.path.join(stats_dir, 'stats.csv'), 'w') as f:
        f.write(f"numeric_columns: {', '.join(numeric_columns)}\n\n")
        f.write(f"non_numeric_columns: {', '.join(nonnumeric_columns)}\n\n")        
        f.write("Summary Statistics:\n")
        summary_statistics.to_csv(f)
        f.write("\n\n")

        f.write(f"Total number of rows: {total_rows}\n\n")

        f.write("Counts of Non-Null and Null Values:\n")
        for column in df.columns:
            non_null = non_null_counts[column]
            null = null_counts[column]
            f.write(f"{column}: {non_null} non-null values, {null} null values out of {total_rows} rows\n")
        f.write("\n\n")

        f.write("Value Counts for Each Column:\n")
        for column in df.columns:
            f.write(f"\nColumn: {column}\n")
            value_counts = df[column].value_counts().head(10)
            value_counts.to_csv(f, header=True)
        f.write("\n\n")

        f.write("Unique Value Counts:\n")
        unique_counts.to_csv(f)
        f.write("\n\n")

        if class_imbalance is not None:
            f.write("Class Imbalance:\n")
            class_imbalance.to_csv(f)
            f.write("\n\n")

        f.write("Correlation Matrix:\n")
        correlation_matrix.to_csv(f)
        f.write("\n\n")

        f.write("Label Co-occurrence Matrix:\n")
        label_cooccurrence_matrix.to_csv(f)
        f.write("\n\n")
    log_time("Saving statistics to CSV")
    
    print("Saving correlation matrix heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(os.path.join(stats_dir, 'correlation_matrix_heatmap.png'))
    log_time("Saving correlation matrix heatmap")

    print("Saving missing values heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig(os.path.join(stats_dir, 'missing_values_heatmap.png'))
    log_time("Saving missing values heatmap")
    
    print("Generating distribution plots for numeric columns...")
    for column in tqdm(numeric_columns, desc="Distribution Plots"):
        plt.figure()
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(os.path.join(stats_dir, f'distribution_{column}.png'))
    log_time("Generating distribution plots for numeric columns")
    
    print("Generating box plots for numeric columns to identify outliers...")
    for column in tqdm(numeric_columns, desc="Box Plots"):
        try:
            print(f"Plotting boxplot for {column}...")
            plt.figure()
            sns.boxplot(x=df[column])
            plt.title(f'Box Plot of {column}')
            plt.savefig(os.path.join(stats_dir, f'boxplot_{column}.png'))
        except Exception as e:
            print(f"Skipping boxplot for {column} due to error: {e}")
    log_time("Generating box plots for numeric columns to identify outliers")

    
    print("Generating pair plot for a subset of numeric columns...")
    # Select highly correlated pairs (e.g., correlation coefficient > 0.7 or < -0.7)
    high_corr_pairs = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]
    # Get the names of the highly correlated columns
    high_corr_columns = high_corr_pairs.columns[high_corr_pairs.any()].tolist()
    # Select top N columns
    # Determine the number of columns per plot
    num_columns_per_plot = 10
    num_plots = int(np.ceil(len(high_corr_columns) / num_columns_per_plot))

    # Generate pair plots for each subset of columns
    for i in range(num_plots):
        start_idx = i * num_columns_per_plot
        end_idx = min(start_idx + num_columns_per_plot, len(high_corr_columns))
        subset_columns = high_corr_columns[start_idx:end_idx]
        
        sns.pairplot(df[subset_columns])
        plt.savefig(os.path.join(stats_dir, f'pairplot_{i + 1}.png'))
        plt.close()  # Close the figure to avoid memory issues

    log_time("Generating pair plots for subsets of numeric columns")
    
    for target_column in label_columns:
        print(f"Processing target column: {target_column}")
        if target_column in df.columns and target_column in numeric_columns:
            print("Calculating correlation with target variable...")
            correlation_with_target = df[numeric_columns].corr()[target_column].sort_values(ascending=False)
            correlation_with_target.to_csv(os.path.join(stats_dir, 'correlation_with_target.csv'))

            print("Saving correlation with target variable plot...")
            plt.figure(figsize=(10, 8))
            sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index)
            plt.title(f'Correlation with {target_column}')
            plt.savefig(os.path.join(stats_dir, f'correlation_with_{target_column}.png'))
            log_time("Calculating and saving correlation with target variable")

def bofb(name,thePlan,theTarget,theContext):
    isDebug = True
    mlOutputPath = "/oncmloutput"
    pheaders = {'Content-type': 'application/xml'}
    print(fhir_url+'/$bundleofbundles?patient='+name+'&plan='+thePlan+'&context='+theContext)
    bofb_response = requests.get(fhir_url+'/$bundleofbundles?patient='+name+'&plan='+thePlan +'&context='+theContext,  headers=pheaders)     
    if bofb_response.ok:
        print("bundleofbundles successfully!")
        if isDebug:
            with open(mlOutputPath+"/"+ theContext+ thePlan +name + ".xml", "w+") as text_file:
                text_file.write(bofb_response.text)        
        headers = {
                'Content-Type': 'application/xml'
            }
        print("Start transformationResult!" + "http://localhost:8282/mdmi/transformation/byvalue/?source=FHIRR4JSON.MasterBundle&target=ONCML."+theTarget)    
        transformationResult = requests.request("POST", "http://localhost:8282/mdmi/transformation/byvalue/?source=FHIRR4JSON.MasterBundle&target=ONCML."+theTarget, headers=headers, data=bofb_response.content)
        print("End transformationResult!" + "http://localhost:8282/mdmi/transformation/byvalue/?source=FHIRR4JSON.MasterBundle&target=ONCML."+theTarget)    
        print("END transformationResult!")    

        if transformationResult.ok:
            print("2222 bundleofbundles successfully! " + mlOutputPath+"/"+ theContext+thePlan + name + ".csv")
            with open(mlOutputPath+"/"+  theContext+thePlan + name + ".csv", "w+") as text_file:
                text_file.write(str(transformationResult.content, encoding='utf-8'))
        else:
            print("Error transformationResult! " + transformationResult.status_code)    

# [rest of code]
def thread_function(name,thePlan,theTarget):
    logging.info(thePlan)
    logging.info(theTarget)
    logging.info("Thread %s %s %s: starting", name,thePlan,theTarget)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":

 

    maxpatients = int(os.environ.get('MAXPATIENTS','1000000'))

 
    runqa = bool(os.environ.get('RUNQA', 'False'))

    env_var = os.getenv('RUNQA', 'False')

    runqa = env_var.lower() in ['true', '1', 'yes']


    print("maxpatients ",maxpatients)    
    print("runqa ",runqa)    

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")


    fhir_url = "http://localhost:8080/fhir"                    
    xpheaders = {'Content-type': 'application/json'}
    xbofb_response = requests.get(fhir_url+'/Patient',  headers=xpheaders) 

    patient_list =  getPatientIds(maxpatients)
   
    # json_object = json.loads(xbofb_response.text)
    # for _entry in json_object["entry"]: 
        # print(_entry["resource"]["id"])
        # patient_list.append(_entry["resource"]["id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(bofb, patient_list,repeat('cqlplan'),repeat('PHASE2'),repeat('ENCOUNTER'))

    print("DONE DONE transformationResult!")        

    destination_file = open('/oncmloutput/' + 'cqlplan' + 'file.csv', 'w', newline='')
    csv_writer = csv.writer(destination_file)
    isFirst = True

    combine_data('cqlplan', 'ENCOUNTER', patient_list, csv_writer)

    if runqa:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(bofb, patient_list,repeat('qaplan'),repeat('ANALYSIS'),repeat('ENCOUNTER'))   

        destination_file = open('/oncmloutput/' + 'qaplan' + 'file.csv', 'w', newline='')
        csv_writer = csv.writer(destination_file)
        isFirst = True


        for patient in patient_list:
            print(patient)

            file1 = open('/oncmloutput'+"/ENCOUNTERqaplan"+ patient + ".csv", 'r')
            print('Parsing /oncmloutput'+"/ENCOUNTERqaplan"+ patient + ".csv")       
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


        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(bofb, patient_list,repeat('qaplan'),repeat('ANALYSIS'),repeat('PATIENT'))  

        destination_file = open('/oncmloutput/' + 'qaplanperpatient' + 'file.csv', 'w', newline='')
        csv_writer = csv.writer(destination_file)
        isFirst = True


        for patient in patient_list:
            print(patient)

            file1 = open('/oncmloutput'+"/PATIENTqaplan"+ patient + ".csv", 'r')
            print('Parsing /oncmloutput'+"/PATIENTqaplan"+ patient + ".csv")       
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

    runstats()    
