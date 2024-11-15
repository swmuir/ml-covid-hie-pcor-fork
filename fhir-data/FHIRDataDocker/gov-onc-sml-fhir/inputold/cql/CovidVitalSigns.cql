library CovidVitalSigns version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.1.000'
include FHIRCommon version '1.0.0' called FC
include USCoreFHIRElements version '1.0.0' called UE
include CovidConcepts called Cx

context Patient

/// Report QA analysis with count of resources, no PHI.
define ReportVitalSignsQA: 
{
  "All Vital Signs Count": Count("All Vital Signs"),
  "Blood Pressure Count": Count("Blood Pressure"),
  "Body Weight Count": Count("Body Weight"),
  "Body Height Count": Count("Body Height"),
  "BMI Count": Count("Body Mass Index"),
  "Body Temperature Count": Count("Body Temperature"),
  "Heart Rate Count": Count("Heart Rate"),
  "Oxygen Saturation Count": Count("Oxygen Saturation"),
  "Respiratory Rate Count": Count("Respiratory Rate"),
  "Vital Signs Without Preferred Code Count": Count("Vital Signs Without Preferred Code")
}

// Most recent FHIR Observation for each vital sign data element.
define ReportVitalSigns: {
  "Systolic Blood Pressure": ToString("Blood Pressure".mostRecent().systolicValue().value),
  "Diastolic Blood Pressure": ToString("Blood Pressure".mostRecent().diastolicValue().value),
  "Body Weight": "Body Weight".mostRecentResult(),
  "Body Height": "Body Height".mostRecentResult(),
  "BMI": "Body Mass Index".mostRecentResult(),
  "Body Temperature": "Body Temperature".mostRecentResult(),
  "Heart Rate": "Heart Rate".mostRecentResult(),
  "Oxygen Saturation": "Oxygen Saturation".mostRecentResult(),
  "Respiratory Rate": "Respiratory Rate".mostRecentResult()
}

define "Vital Signs Without Preferred Code":
  UE."All Vital Signs" obs
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

define fluent function systolicValue(o Observation): 
  if o.code ~ Cx."Blood pressure" then
    (singleton from (o.component C where (C.code ~ Cx."Systolic blood pressure"))).value as FHIR.Quantity
  else
    null

define fluent function diastolicValue(o Observation): 
  if o.code ~ Cx."Blood pressure" then
    (singleton from (o.component C where (C.code ~ Cx."Diastolic blood pressure"))).value as FHIR.Quantity
  else
    null

define "All Vital Signs":
  UE."All Vital Signs".resulted()

define "Blood Pressure":
  "All Vital Signs" obs
    where obs.code ~ Cx."Blood pressure"

define "Body Temperature":
  "All Vital Signs" obs
    where obs.code ~ Cx."Body temperature"

define "Body Height":
  "All Vital Signs" obs
    where obs.code ~ Cx."Body height"

define "Body Weight":
  "All Vital Signs" obs
    where obs.code ~ Cx."Body weight"

define "Body Mass Index":
  "All Vital Signs" obs
    where obs.code ~ Cx."Body mass index"

define "Heart Rate":
  "All Vital Signs" obs
    where obs.code ~ Cx."Heart rate"

define "Oxygen Saturation":
  "All Vital Signs" obs
    where obs.code ~ Cx."Oxygen saturation"

define "Respiratory Rate":
  "All Vital Signs" obs
    where obs.code ~ Cx."Respiratory rate"
