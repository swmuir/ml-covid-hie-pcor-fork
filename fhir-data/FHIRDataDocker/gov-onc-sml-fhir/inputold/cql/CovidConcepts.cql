library CovidConcepts version '1.0.0'

// ## Code Systems ##

codesystem "LOINC": 'http://loinc.org'
codesystem "SNOMEDCT": 'http://snomed.info/sct'
codesystem "Race": 'urn:oid:2.16.840.1.113883.6.238'

// ## Value sets ##

// ******************************
// Condition value sets
// ******************************

valueset "Acute COVID 19 Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1099.58'
valueset "Post Acute Sequelae of Sars Cov 2 infection (PASC)": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1530'
valueset "Acute Myocardial Infarction": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.666.5.3011'
valueset "Ischemic Heart Disease": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.614'
valueset "Acute Coronary Syndromes": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.3157.2000.10'
valueset "Bronchiectasis Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1463'
valueset "Hypertension, Pulmonary hypertension": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1484'
valueset "Pulmonary embolism diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1481'
valueset "Cerebrovascular disease diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1549'
valueset "Chronic kidney disease all stages (1 through 5)": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.159'
valueset "Cirrhosis or other liver disease": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1029.63'
valueset "Chronic obstructive pulmonary disease (COPD) Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1466'
valueset "Type 1 Diabetes": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.464.1003.103.12.1020'
valueset "Type II Diabetes": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.464.1003.103.12.1021'
valueset "Heart Failure Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1543'
valueset "Coronary Artery Disease No MI": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.526.3.369'
valueset "Cardiomyopathy": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.579'
valueset "Interstitial lung disease diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1469'
valueset "Ever tobacco smoker": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1078.504'
valueset "Tuberculosis Disorders": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.43'
valueset "Obesity Conditions": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.36'
valueset "Pregnancy": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1566'
valueset "Depression Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.600.145'
valueset "Schizophrenia": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.464.1003.105.12.1205'
valueset "HIV": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.464.1003.120.12.1003'
valueset "Dementia and Related Intracranial Pathologies": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.3157.4043'
valueset "Overweight or Obese": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1047.502'
valueset "Sickle cell disease and related blood disorders": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1029.35'
valueset "Substance abuse": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.464.1003.106.12.1004'
valueset "Cystic fibrosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.3.464.1003.102.12.1002'
valueset "Thalassemia": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.773'
valueset "Asthma diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1472'
valueset "Hypertension, Primary and Secondary Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1547'
valueset "Chronic Pain Conditions": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.76'
valueset "PostTraumatic stress disorder PTSD": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.103'
valueset "Myalgic Encephalomyelitis or Chronic Fatigue Syndrome": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.675'
valueset "Postural tachycardia syndrome (POTS) Diagnosis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1488'
valueset "Ehlers-Danlos syndrome": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1505'
valueset "Stroke": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.618'
valueset "Food insecurity diagnoses": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1247.17'
valueset "Transportation insecurity": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1247.26'

valueset "Down Syndrome (Disorders) (SNOMED)": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1146.2124'
valueset "Down Syndrome (Disorders) (ICD10CM)": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1146.2125'
valueset "Housing Instability Diagnoses": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1247.24'
valueset "Inadequate Housing Diagnoses": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1247.48'
valueset "Homelessness Diagnoses": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1247.18'

// ******************************
// Lab result value sets
// ******************************

// Value sets from VSAC
valueset "COVID19 Lab Tests": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1580'
valueset "Hemoglobin A1C": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.119'
valueset "Blood Urea Nitrogen": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.113'
valueset "Bilirubin lab test": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1248.223'
valueset "Troponin Lab Test": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.13.190.5.3'
valueset "Ferritin": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113883.13.190.5.3'
valueset "Glucose Tolerance Testing": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1034.360'
valueset "Cerebral Spinal Fluid (CSF) Analysis": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1573'

// Value sets defined by project team in spreadsheets
valueset "Arterial Blood Gas": 'http://emiadvisors.net/fhir/ValueSet/arterial-blood-gas'
valueset "Comprehensive Metabolic Panel": 'http://emiadvisors.net/fhir/ValueSet/comprehensive-metabolic-panel'
valueset "Calcium in Blood, Serum or Plasma": 'http://emiadvisors.net/fhir/ValueSet/calcium-blood-serum-plasma'
valueset "Magnesium in Blood, Serum or Plasma": 'http://emiadvisors.net/fhir/ValueSet/magnesium-blood-serum-plasma'
valueset "Magnesium in Urine": 'http://emiadvisors.net/fhir/ValueSet/magnesium-urine'
valueset "Chloride, Urine": 'http://emiadvisors.net/fhir/ValueSet/chloride-urine'
valueset "Chloride, Blood, Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/chloride-blood-serum-plasma'
valueset "Creatinine, Urine": 'http://emiadvisors.net/fhir/ValueSet/creatinine-urine'
valueset "Creatinine, Blood, Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/creatinine-blood-serum-plasma'
valueset "Phosphate Blood, Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/phosphate-blood-serum-plasma'

// Spreadsheets 11-20
valueset "Coagulation Assay": 'http://emiadvisors.net/fhir/ValueSet/coagulation-assay'
valueset "Complete Blood Count": 'http://emiadvisors.net/fhir/ValueSet/complete-blood-count'
valueset "Creatine Kinase Blood, Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/creatine-kinase-blood-serum-plasma'
valueset "D Dimer Test": 'http://emiadvisors.net/fhir/ValueSet/d-dimer-test'
valueset "Electrolytes Panel Blood, Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/electrolytes-panel-blood-serum-plasma'
valueset "Inflammatory Markers (CRP) Blood, Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/inflammatory-markers-blood-serum-plasma'
valueset "Lipid Serum, or Plasma": 'http://emiadvisors.net/fhir/ValueSet/lipid-serum-plasma'
valueset "Sputum Culture": 'http://emiadvisors.net/fhir/ValueSet/sputum-culture'
valueset "Urine Collection 24 Hours": 'http://emiadvisors.net/fhir/ValueSet/urine-collection-24-hours'
valueset "Urine Routine": 'http://emiadvisors.net/fhir/ValueSet/urine-routine'


// ******************************
// Symptom value sets
// ******************************

valueset "Brain Fog": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1377'
valueset "Impaired Sense of Smell": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1396'
valueset "Impaired Sense of Taste": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1385'
valueset "Post Exertional Malaise": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1457'
valueset "Respiratory Distress": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.1414'


// ******************************
// Procedure value sets
// ******************************

valueset "Echocardiogram findings": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1222.876'
valueset "CT Angiography of Chest": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1147.155'

// ******************************
// Medication value sets
// ******************************
// valueset "ACEis and ARBs": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1213.11'
// valueset "Erythropoiesis Stimulating Agent": 'http://cts.nlm.nih.gov/fhir/ValueSet/2.16.840.1.113762.1.4.1196.307'

// ******************************
// ## Individual codes ##
// ******************************

// list individual codes from code systems used directly in the CQL logic. Individual codes are used when there is
// a single code from a particular vocabulary standard used to represent a clinical concept. It is considered
// best-practice not to create value sets containing a single code.

code "Black or African American": '2054-5' from "Race" display 'Black or African American'
code "White": '2106-3' from "Race" display 'White'

code "Left (qualifier value)": '7771000' from "SNOMEDCT" display 'Left (qualifier value)'
code "Right (qualifier value)": '24028007' from "SNOMEDCT" display 'Right (qualifier value)'

code "Tobacco smoking status": '72166-2' from "LOINC" display 'Tobacco smoking status'

code "Blood pressure": '85354-9' from "LOINC" display 'Blood pressure systolic and diastolic'
code "Systolic blood pressure": '8480-6' from "LOINC" display 'Systolic blood pressure'
code "Diastolic blood pressure": '8462-4' from "LOINC" display 'Diastolic blood pressure'
code "Heart rate": '8867-4' from "LOINC" display 'Heart rate'
code "Oxygen saturation": '2708-6' from "LOINC" display 'Oxygen saturation'
code "Pulse oximetry": '59408-5' from LOINC display 'Oxygen saturation in Arterial blood by Pulse oximetry'
code "Respiratory rate": '9279-1' from "LOINC" display 'Respiratory rate'
code "Body temperature": '8310-5' from "LOINC" display 'Body temperature'
code "Body height": '8302-2' from "LOINC" display 'Body height'
code "Body weight": '29463-7' from "LOINC" display 'Body weight'
code "Body mass index": '39156-5' from "LOINC" display 'Body mass index'
code "Inhaled oxygen flow rate": '3151-8' from "LOINC" display 'Inhaled oxygen flow rate'
