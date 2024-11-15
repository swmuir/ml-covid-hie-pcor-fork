library CovidLabResults version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.1.000'
include FHIRCommon version '1.0.0' called FC
include USCoreFHIRElements version '1.0.0' called UE
include CovidConcepts called Cx

context Patient

/// Report QA analysis with count of resources, no PHI.
define ReportLabResultsQA: 
{
  "Lab Results Count": Count(UE."All Laboratory Results"),
  "Lab Results Without Preferred Code Count": Count("Lab Results Without Preferred Code"),
  "COVID19 Lab Tests Count": Count("COVID19 Lab Tests"),
  "Hemoglobin A1C Count": Count("Hemoglobin A1C"),
  "Blood Urea Nitrogen Count": Count("Blood Urea Nitrogen"),
  "Bilirubin lab test Count": Count("Bilirubin lab test"),
  "Troponin Lab Test Count": Count("Troponin Lab Test"),
  "Ferritin Count": Count("Ferritin"),
  "Glucose Tolerance Testing Count": Count("Glucose Tolerance Testing"),
  "Cerebral Spinal Fluid (CSF) Analysis Count": Count("Cerebral Spinal Fluid (CSF) Analysis"),
  "Arterial Blood Gas Count": Count("Arterial Blood Gas"),
  "Comprehensive Metabolic Panel Count": Count("Comprehensive Metabolic Panel"),
  "Calcium in Blood, Serum or Plasma Count": Count("Calcium in Blood, Serum or Plasma"),
  "Magnesium in Blood, Serum or Plasma Count": Count("Magnesium in Blood, Serum or Plasma"),
  "Magnesium in Urine Count": Count("Magnesium in Urine"),
  "Chloride, Urine Count": Count("Chloride, Urine"),
  "Chloride, Blood, Serum, or Plasma Count": Count("Chloride, Blood, Serum, or Plasma"),
  "Creatinine, Urine Count": Count("Creatinine, Urine"),
  "Creatinine, Blood, Serum, or Plasma Count": Count("Creatinine, Blood, Serum, or Plasma"),
  "Phosphate Blood, Serum, or Plasma Count": Count("Phosphate Blood, Serum, or Plasma"),
  "Coagulation Assay Count": Count("Coagulation Assay"),
  "Complete Blood Count Count": Count("Complete Blood Count"),
  "Creatine Kinase Blood, Serum, or Plasma Count": Count("Creatine Kinase Blood, Serum, or Plasma"),
  "D Dimer Test Count": Count("D Dimer Test"),
  "Electrolytes Panel Blood, Serum, or Plasma Count": Count("Electrolytes Panel Blood, Serum, or Plasma"),
  "Inflammatory Markers (CRP) Blood, Serum, or Plasma Count": Count("Inflammatory Markers (CRP) Blood, Serum, or Plasma"),
  "Lipid Serum, or Plasma Count": Count("Lipid Serum, or Plasma"),
  "Sputum Culture Count": Count("Sputum Culture"),
  "Urine Collection 24 Hours Count": Count("Urine Collection 24 Hours"),
  "Urine Routine Count": Count("Urine Routine")
}

// Most recent FHIR Observation for each laboratory data element.
define ReportLabResults: {
  "COVID19 Lab Tests": "COVID19 Lab Tests".mostRecentResult(),
  "Hemoglobin A1C": "Hemoglobin A1C".mostRecentResult(),
  "Blood Urea Nitrogen": "Blood Urea Nitrogen".mostRecentResult(),
  "Bilirubin lab test": "Bilirubin lab test".mostRecentResult(),
  "Troponin Lab Test": "Troponin Lab Test".mostRecentResult(),
  "Ferritin": "Ferritin".mostRecentResult(),
  "Glucose Tolerance Testing": "Glucose Tolerance Testing".mostRecentResult(),
  "Cerebral Spinal Fluid (CSF) Analysis": "Cerebral Spinal Fluid (CSF) Analysis".mostRecentResult(),

  "Arterial Blood Gas": "Arterial Blood Gas".mostRecentResult(),
  "Comprehensive Metabolic Panel": "Comprehensive Metabolic Panel".mostRecentResult(),
  "Calcium in Blood, Serum or Plasma": "Calcium in Blood, Serum or Plasma".mostRecentResult(),
  "Magnesium in Blood, Serum or Plasma": "Magnesium in Blood, Serum or Plasma".mostRecentResult(),
  "Magnesium in Urine": "Magnesium in Urine".mostRecentResult(),
  "Chloride, Urine": "Chloride, Urine".mostRecentResult(),
  "Chloride, Blood, Serum, or Plasma": "Chloride, Blood, Serum, or Plasma".mostRecentResult(),
  "Creatinine, Urine": "Creatinine, Urine".mostRecentResult(),
  "Creatinine, Blood, Serum, or Plasma": "Creatinine, Blood, Serum, or Plasma".mostRecentResult(),
  "Phosphate Blood, Serum, or Plasma": "Phosphate Blood, Serum, or Plasma".mostRecentResult(),
  "Coagulation Assay": "Coagulation Assay".mostRecentResult(),
  "Complete Blood Count": "Complete Blood Count".mostRecentResult(),
  "Creatine Kinase Blood, Serum, or Plasma": "Creatine Kinase Blood, Serum, or Plasma".mostRecentResult(),
  "D Dimer Test": "D Dimer Test".mostRecentResult(),
  "Electrolytes Panel Blood, Serum, or Plasma": "Electrolytes Panel Blood, Serum, or Plasma".mostRecentResult(),
  "Inflammatory Markers (CRP) Blood, Serum, or Plasma": "Inflammatory Markers (CRP) Blood, Serum, or Plasma".mostRecentResult(),
  "Lipid Serum, or Plasma": "Lipid Serum, or Plasma".mostRecentResult(),
  "Sputum Culture": "Sputum Culture".mostRecentResult(),
  "Urine Collection 24 Hours": "Urine Collection 24 Hours".mostRecentResult(),
  "Urine Routine": "Urine Routine".mostRecentResult()
}

define "Lab Results Without Preferred Code":
  UE."All Laboratory Results" obs
    where obs.preferredCode() is null

// Returns a System.Code with sytem and code parameters
define fluent function preferredCode(observation Observation):
  First(
    observation.code.coding c
      let preferredCode: c.preferredCoding()
      where preferredCode.code is not null
      return preferredCode
  )

define fluent function preferredCoding(c FHIR.Coding):
  case
    when c.system.value = 'http://loinc.org' then
      System.Code { system: 'http://loinc.org', code: c.code.value, display: c.display.value }
    else
      null
  end

define fluent function mostRecentResult(observations List<Observation>):
  FC.ResultText(observations.mostRecent())

// Lab Result FHIR Observations

define "All Laboratory Results":
  UE."All Laboratory Results".resulted()

define "COVID19 Lab Tests":
  "All Laboratory Results" lab
    where lab.code in Cx."COVID19 Lab Tests"

define "Hemoglobin A1C":
  "All Laboratory Results" lab
    where lab.code in Cx."Hemoglobin A1C"

define "Blood Urea Nitrogen":
  "All Laboratory Results" lab
    where lab.code in Cx."Blood Urea Nitrogen"

define "Bilirubin lab test":
  "All Laboratory Results" lab
    where lab.code in Cx."Bilirubin lab test"

define "Troponin Lab Test":
  "All Laboratory Results" lab
    where lab.code in Cx."Troponin Lab Test"

define "Ferritin":
  "All Laboratory Results" lab
    where lab.code in Cx."Ferritin"

define "Glucose Tolerance Testing":
  "All Laboratory Results" lab
    where lab.code in Cx."Glucose Tolerance Testing"

define "Cerebral Spinal Fluid (CSF) Analysis":
  "All Laboratory Results" lab
    where lab.code in Cx."Cerebral Spinal Fluid (CSF) Analysis"


define "Arterial Blood Gas":
  "All Laboratory Results" lab
    where lab.code in Cx."Arterial Blood Gas"

define "Comprehensive Metabolic Panel":
  "All Laboratory Results" lab
    where lab.code in Cx."Comprehensive Metabolic Panel"

define "Calcium in Blood, Serum or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Calcium in Blood, Serum or Plasma"

define "Magnesium in Blood, Serum or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Magnesium in Blood, Serum or Plasma"

define "Magnesium in Urine":
  "All Laboratory Results" lab
    where lab.code in Cx."Magnesium in Urine"

define "Chloride, Urine":
  "All Laboratory Results" lab
    where lab.code in Cx."Chloride, Urine"

define "Chloride, Blood, Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Chloride, Blood, Serum, or Plasma"

define "Creatinine, Urine":
  "All Laboratory Results" lab
    where lab.code in Cx."Creatinine, Urine"

define "Creatinine, Blood, Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Creatinine, Blood, Serum, or Plasma"

define "Phosphate Blood, Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Phosphate Blood, Serum, or Plasma"

define "Coagulation Assay":
  "All Laboratory Results" lab
    where lab.code in Cx."Coagulation Assay"

define "Complete Blood Count":
  "All Laboratory Results" lab
    where lab.code in Cx."Complete Blood Count"

define "Creatine Kinase Blood, Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Creatine Kinase Blood, Serum, or Plasma"

define "D Dimer Test":
  "All Laboratory Results" lab
    where lab.code in Cx."D Dimer Test"

define "Electrolytes Panel Blood, Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Electrolytes Panel Blood, Serum, or Plasma"

define "Inflammatory Markers (CRP) Blood, Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Inflammatory Markers (CRP) Blood, Serum, or Plasma"

define "Lipid Serum, or Plasma":
  "All Laboratory Results" lab
    where lab.code in Cx."Lipid Serum, or Plasma"

define "Sputum Culture":
  "All Laboratory Results" lab
    where lab.code in Cx."Sputum Culture"

define "Urine Collection 24 Hours":
  "All Laboratory Results" lab
    where lab.code in Cx."Urine Collection 24 Hours"

define "Urine Routine":
  "All Laboratory Results" lab
    where lab.code in Cx."Urine Routine"
