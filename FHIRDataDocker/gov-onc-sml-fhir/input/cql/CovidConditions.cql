library CovidConditions version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.1.000'
include FHIRCommon version '1.0.0' called FC
include USCoreFHIRElements version '1.0.0' called UE
include CovidConcepts called Cx

context Patient

/// Report QA analysis with count of resources, no PHI.
define ReportConditionsQA: 
{
  "All Conditions Count": Count(UE."All Conditions"),
  "Active Problem List Count": Count(UE."Active Problem List Items"),
  "Inactive Problem List Count": Count("Inactive Problem List"),
  "Encounter Diagnoses Count": Count(UE."All Encounter Diagnoses"),
  "Uncategorized Conditions Count": Count("Uncategorized Conditions"),
  "Conditions Without Preferred Code Count": Count("Conditions Without Preferred Code"),
  "Conditions With SNOMED Code Count": Count(UE."All Conditions".snomedCodes()),
  "Conditions With ICD-10 Code Count": Count(UE."All Conditions".icd10Codes())
}

/// Report boolean values for input to ML model. Also list of codes for active problem list items.
define ReportConditions: 
{
  "Acute COVID 19 Diagnosis": (Count("Acute COVID 19 Diagnosis") > 0).toInteger(),
  "Post Acute Sequelae of Sars Cov 2 infection (PASC)": (Count("Post Acute Sequelae of Sars Cov 2 infection (PASC)") > 0).toInteger(),
  "COVID 19 Diagnosis": (Count("COVID 19 Diagnosis") > 0).toInteger(),
  "Acute Myocardial Infarction": (Count("Acute Myocardial Infarction") > 0).toInteger(),
  "Ischemic Heart Disease": (Count("Ischemic Heart Disease") > 0).toInteger(),
  "Acute Coronary Syndromes": (Count("Acute Coronary Syndromes") > 0).toInteger(),
  "Bronchiectasis Diagnosis": (Count("Bronchiectasis Diagnosis") > 0).toInteger(),
  "Hypertension, Pulmonary hypertension": (Count("Hypertension, Pulmonary hypertension") > 0).toInteger(),
  "Pulmonary embolism diagnosis": (Count("Pulmonary embolism diagnosis") > 0).toInteger(),
  "Cerebrovascular disease diagnosis": (Count("Cerebrovascular disease diagnosis") > 0).toInteger(),
  "Chronic kidney disease all stages (1 through 5)": (Count("Chronic kidney disease all stages (1 through 5)") > 0).toInteger(),
  "Cirrhosis or other liver disease": (Count("Cirrhosis or other liver disease") > 0).toInteger(),
  "Chronic obstructive pulmonary disease (COPD) Diagnosis": (Count("Chronic obstructive pulmonary disease (COPD) Diagnosis") > 0).toInteger(),
  "Type 1 Diabetes": (Count("Type 1 Diabetes") > 0).toInteger(),
  "Type II Diabetes": (Count("Type II Diabetes") > 0).toInteger(),
  "Diabetes (All Types)": (Count("Diabetes (All Types)") > 0).toInteger(),
  "Heart Failure Diagnosis": (Count("Heart Failure Diagnosis") > 0).toInteger(),
  "Coronary Artery Disease No MI": (Count("Coronary Artery Disease No MI") > 0).toInteger(),
  "Cardiomyopathy": (Count("Cardiomyopathy") > 0).toInteger(),
  "Interstitial lung disease diagnosis": (Count("Interstitial lung disease diagnosis") > 0).toInteger(),
  "Ever tobacco smoker": (Count("Ever tobacco smoker") > 0).toInteger(),
  "Tuberculosis Disorders": (Count("Tuberculosis Disorders") > 0).toInteger(),
  "Obesity Conditions": (Count("Obesity Conditions") > 0).toInteger(),
  "Pregnancy": (Count("Pregnancy") > 0).toInteger(),
  "Depression Diagnosis": (Count("Depression Diagnosis") > 0).toInteger(),
  "Schizophrenia": (Count("Schizophrenia") > 0).toInteger(),
  "HIV": (Count("HIV") > 0).toInteger(),
  "Dementia and Related Intracranial Pathologies": (Count("Dementia and Related Intracranial Pathologies") > 0).toInteger(),
  "Overweight or Obese": (Count("Overweight or Obese") > 0).toInteger(),
  "Sickle cell disease and related blood disorders": (Count("Sickle cell disease and related blood disorders") > 0).toInteger(),
  "Substance abuse": (Count("Substance abuse") > 0).toInteger(),
  "Cystic fibrosis": (Count("Cystic fibrosis") > 0).toInteger(),
  "Thalassemia": (Count("Thalassemia") > 0).toInteger(),
  "Asthma diagnosis": (Count("Asthma diagnosis") > 0).toInteger(),
  "Hypertension, Primary and Secondary Diagnosis": (Count("Hypertension, Primary and Secondary Diagnosis") > 0).toInteger(),
  "Chronic Pain Conditions": (Count("Chronic Pain Conditions") > 0).toInteger(),
  "PostTraumatic stress disorder PTSD": (Count("PostTraumatic stress disorder PTSD") > 0).toInteger(),
  "Myalgic Encephalomyelitis or Chronic Fatigue Syndrome": (Count("Myalgic Encephalomyelitis or Chronic Fatigue Syndrome") > 0).toInteger(),
  "Postural tachycardia syndrome (POTS) Diagnosis": (Count("Postural tachycardia syndrome (POTS) Diagnosis") > 0).toInteger(),
  "Ehlers-Danlos syndrome": (Count("Ehlers-Danlos syndrome") > 0).toInteger(),
  "Stroke": (Count("Stroke") > 0).toInteger(),
  "Down Syndrome": (Count("Down Syndrome") > 0).toInteger(),
  "Food insecurity diagnoses": (Count("Food insecurity diagnoses") > 0).toInteger(),
  "Housing insecurity": (Count("Housing insecurity") > 0).toInteger(),
  "Transportation insecurity": (Count("Transportation insecurity") > 0).toInteger(),

  "Problem List Codes": UE."All Problem List Items".preferredCodes(),
  "Condition Codes": UE."All Conditions".preferredCodes(),
  "SNOMED Codes": UE."All Conditions".snomedCodes(),
  "ICD-10 Codes": UE."All Conditions".icd10Codes()
}

define fluent function toInteger(boolValue boolean):
  if boolValue is true then 1
  else 0

define "Inactive Problem List":
  UE."All Problem List Items" condition
    where condition.isActive() is not true
      or condition.isConfirmed() is not true

define "Uncategorized Conditions":
  UE."All Conditions" condition
    where condition.isProblemListItem() is not true
      and condition.isEncounterDiagnosis() is not true

define "Conditions Without Preferred Code":
  UE."All Conditions" condition
    where condition.preferredCode() is null

// Returns a list of code tokens
define fluent function preferredCodes(conditions List<Condition>):
  conditions condition
    let preferredCode: condition.preferredCode()
    where preferredCode.code is not null
    return preferredCode.code
  
// Returns a list of code tokens
define fluent function snomedCodes(conditions List<Condition>):
  conditions condition
    let preferredCode: condition.snomedCode()
    where preferredCode.code is not null
    return preferredCode.code
  
// Returns a list of code tokens
define fluent function icd10Codes(conditions List<Condition>):
  conditions condition
    let preferredCode: condition.icd10Code()
    where preferredCode.code is not null
    return preferredCode.code
  
// Returns a System.Code with sytem and code parameters
define fluent function preferredCode(condition Condition):
  Coalesce((condition.code.coding) c return c.preferredConditionCoding())

define fluent function snomedCode(condition Condition):
  Coalesce((condition.code.coding) c return c.snomedCoding())

define fluent function icd10Code(condition Condition):
  Coalesce((condition.code.coding) c return c.icd10Coding())


// define fluent function preferredCode2(condition Condition):
//   First(
//     condition.code.coding c
//       let preferredCode: PreferredConditionCoding(c)
//       where preferredCode.code is not null
//       return preferredCode
//   )

define fluent function preferredConditionCoding(c FHIR.Coding):
  case
    when c.system.value = 'http://snomed.info/sct' or c.system.value = 'urn:oid:2.16.840.1.113883.6.96' then
      System.Code { system: 'http://snomed.info/sct', code: c.code.value, display: c.display.value }
    when c.system.value = 'http://hl7.org/fhir/sid/icd-10-cm' or c.system.value = 'urn:oid:2.16.840.1.113883.6.90' then
      System.Code { system: 'http://hl7.org/fhir/sid/icd-10-cm', code: c.code.value, display: c.display.value }
    when c.system.value = 'http://hl7.org/fhir/sid/icd-10' or c.system.value = 'urn:oid:2.16.840.1.113883.6.3' then
      System.Code { system: 'http://hl7.org/fhir/sid/icd-10', code: c.code.value, display: c.display.value }
    when c.system.value = 'http://hl7.org/fhir/sid/icd-9-cm' or c.system.value = 'urn:oid:2.16.840.1.113883.6.103' then
      System.Code { system: 'http://hl7.org/fhir/sid/icd-9-cm', code: c.code.value, display: c.display.value }
    else
      null
  end

define fluent function snomedCoding(c FHIR.Coding):
  case
    when c.system.value = 'http://snomed.info/sct' or c.system.value = 'urn:oid:2.16.840.1.113883.6.96' then
      System.Code { system: 'http://snomed.info/sct', code: c.code.value, display: c.display.value }
    else
      null
  end

define fluent function icd10Coding(c FHIR.Coding):
  case
    when c.system.value = 'http://hl7.org/fhir/sid/icd-10-cm' or c.system.value = 'urn:oid:2.16.840.1.113883.6.90' then
      System.Code { system: 'http://hl7.org/fhir/sid/icd-10-cm', code: c.code.value, display: c.display.value }
    else
      null
  end

//
// Use value sets to categorize active, confirmed Conditions
//

define "Acute COVID 19 Diagnosis":
  ([Condition: Cx."Acute COVID 19 Diagnosis"]).active()

define "Post Acute Sequelae of Sars Cov 2 infection (PASC)":
  ([Condition: Cx."Post Acute Sequelae of Sars Cov 2 infection (PASC)"]).active()

define "COVID 19 Diagnosis":
  "Acute COVID 19 Diagnosis"
  union "Post Acute Sequelae of Sars Cov 2 infection (PASC)"

define "Acute Myocardial Infarction":
  ([Condition: Cx."Acute Myocardial Infarction"]).active()

define "Ischemic Heart Disease":
  ([Condition: Cx."Ischemic Heart Disease"]).active()

define "Acute Coronary Syndromes":
  ([Condition: Cx."Acute Coronary Syndromes"]).active()

define "Bronchiectasis Diagnosis":
  ([Condition: Cx."Bronchiectasis Diagnosis"]).active()

define "Hypertension, Pulmonary hypertension":
  ([Condition: Cx."Hypertension, Pulmonary hypertension"]).active()

define "Pulmonary embolism diagnosis":
  ([Condition: Cx."Pulmonary embolism diagnosis"]).active()

define "Cerebrovascular disease diagnosis":
  ([Condition: Cx."Cerebrovascular disease diagnosis"]).active()

define "Chronic kidney disease all stages (1 through 5)":
  ([Condition: Cx."Chronic kidney disease all stages (1 through 5)"]).active()

define "Cirrhosis or other liver disease":
  ([Condition: Cx."Cirrhosis or other liver disease"]).active()

define "Chronic obstructive pulmonary disease (COPD) Diagnosis":
  ([Condition: Cx."Chronic obstructive pulmonary disease (COPD) Diagnosis"]).active()

define "Type 1 Diabetes":
  ([Condition: Cx."Type 1 Diabetes"]).active()

define "Type II Diabetes":
  ([Condition: Cx."Type II Diabetes"]).active()

define "Diabetes (All Types)":
  "Type 1 Diabetes"
  union "Type II Diabetes"

define "Heart Failure Diagnosis":
  ([Condition: Cx."Heart Failure Diagnosis"]).active()

define "Coronary Artery Disease No MI":
  ([Condition: Cx."Coronary Artery Disease No MI"]).active()

define "Cardiomyopathy":
  ([Condition: Cx."Cardiomyopathy"]).active()

define "Interstitial lung disease diagnosis":
  ([Condition: Cx."Interstitial lung disease diagnosis"]).active()

define "Ever tobacco smoker":
  ([Condition: Cx."Ever tobacco smoker"]).active()

define "Tuberculosis Disorders":
  ([Condition: Cx."Tuberculosis Disorders"]).active()

define "Obesity Conditions":
  ([Condition: Cx."Obesity Conditions"]).active()

define "Pregnancy":
  ([Condition: Cx."Pregnancy"]).active()

define "Depression Diagnosis":
  ([Condition: Cx."Depression Diagnosis"]).active()

define "Schizophrenia":
  ([Condition: Cx."Schizophrenia"]).active()

define "HIV":
  ([Condition: Cx."HIV"]).active()

define "Dementia and Related Intracranial Pathologies":
  ([Condition: Cx."Dementia and Related Intracranial Pathologies"]).active()

define "Overweight or Obese":
  ([Condition: Cx."Overweight or Obese"]).active()

define "Sickle cell disease and related blood disorders":
  ([Condition: Cx."Sickle cell disease and related blood disorders"]).active()

define "Substance abuse":
  ([Condition: Cx."Substance abuse"]).active()

define "Cystic fibrosis":
  ([Condition: Cx."Cystic fibrosis"]).active()

define "Thalassemia":
  ([Condition: Cx."Thalassemia"]).active()

define "Asthma diagnosis":
  ([Condition: Cx."Asthma diagnosis"]).active()

define "Hypertension, Primary and Secondary Diagnosis":
  ([Condition: Cx."Hypertension, Primary and Secondary Diagnosis"]).active()

define "Chronic Pain Conditions":
  ([Condition: Cx."Chronic Pain Conditions"]).active()

define "PostTraumatic stress disorder PTSD":
  ([Condition: Cx."PostTraumatic stress disorder PTSD"]).active()

define "Myalgic Encephalomyelitis or Chronic Fatigue Syndrome":
  ([Condition: Cx."Myalgic Encephalomyelitis or Chronic Fatigue Syndrome"]).active()

define "Postural tachycardia syndrome (POTS) Diagnosis":
  ([Condition: Cx."Postural tachycardia syndrome (POTS) Diagnosis"]).active()

define "Ehlers-Danlos syndrome":
  ([Condition: Cx."Ehlers-Danlos syndrome"]).active()

define "Stroke":
  ([Condition: Cx."Stroke"]).active()

define "Food insecurity diagnoses":
  ([Condition: Cx."Food insecurity diagnoses"]).active()

define "Transportation insecurity":
  ([Condition: Cx."Transportation insecurity"]).active()

define "Down Syndrome (Disorders) (SNOMED)":
  ([Condition: Cx."Down Syndrome (Disorders) (SNOMED)"]).active()

define "Down Syndrome (Disorders) (ICD10CM)":
  ([Condition: Cx."Down Syndrome (Disorders) (ICD10CM)"]).active()

define "Down Syndrome":
  "Down Syndrome (Disorders) (SNOMED)"
  union "Down Syndrome (Disorders) (ICD10CM)"

define "Housing Instability Diagnoses":
  ([Condition: Cx."Housing Instability Diagnoses"]).active()

define "Inadequate Housing Diagnoses":
  ([Condition: Cx."Inadequate Housing Diagnoses"]).active()

define "Homelessness Diagnoses":
  ([Condition: Cx."Homelessness Diagnoses"]).active()

define "Housing insecurity":
  "Housing Instability Diagnoses"
  union "Inadequate Housing Diagnoses"
  union "Homelessness Diagnoses"
/*
// TODO complete this list of all categorized conditions if you want a separate list of other, uncategorized conditions.
define CategorizedConditions: 
  "Acute COVID 19 Diagnosis"
  union "Post Acute Sequelae of Sars Cov 2 infection (PASC)"
  union "Acute Myocardial Infarction"
  union "Ischemic Heart Disease"
  union "Acute Coronary Syndromes"
  union "Hypertension, Primary and Secondary Diagnosis"
  union "Type 1 Diabetes"
  union "Type II Diabetes"
  union "Chronic kidney disease all stages (1 through 5)"

/// Returns a list of Resource.id for all categorized conditions.
define CategorizedConditionIds:
  flatten( CategorizedConditions C
    return { C.id }
  )

/// Returns a list of Conditions that are not classified into CategorizedConditions.
define OtherConditions:
  AllActiveConditions C
    where not (C.id in CategorizedConditionIds)

define ReportOtherConditions: 
  OtherConditions C
    return ReportCondition(C)
  
define function ReportCondition(condition Condition):
  {
    id: condition.id.value,
    code: PreferredConditionConcept(condition.code).code,
    display: DE.ConceptText(condition.code)
  }
*/
