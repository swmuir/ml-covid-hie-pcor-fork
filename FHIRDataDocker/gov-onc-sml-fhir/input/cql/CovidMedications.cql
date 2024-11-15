library CovidMedications version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.1.000'
include FHIRCommon version '1.0.0' called FC
include CovidConcepts called Cx

context Patient

/// Report QA analysis with count of resources, no PHI.
define ReportMedicationsQA: 
{
  "MedicationRequest Count": Count("All Medication Requests"),
  "Medications Without Preferred Code Count": Count("Medications Without Preferred Code")
}

/// Report a list of codes for all procedures, using preferred coding system.
define ReportMedications: 
{
  "Medication Codes": "All Medication Requests".preferredCodes()
}

define "All Medication Requests":
  [MedicationRequest]

define "Medications Without Preferred Code":
  "All Medication Requests" medReq
    where medReq.preferredCode() is null

define fluent function preferredCodes(medRequests List<MedicationRequest>):
  medRequests medReq
    where medReq.preferredCode() is not null
    return medReq.preferredCode().code
  
define fluent function preferredCode(medReq MedicationRequest):
  if medReq.medication is CodeableConcept then
    (medReq.medication as CodeableConcept).preferredMedicationCode()
  else if medReq.medication is Reference then
    ResolveMedication(medReq, (medReq.medication as Reference)).preferredMedCode()
  else
    null

// Returns a Medication resolved from a FHIR Reference, either contained or in the Bundle.
define function ResolveMedication(medReq MedicationRequest, ref Reference):
  Coalesce(
    First([Medication] med
      where med.id = ref.reference.getId()
    ),
    First( medReq.contained med
      where med.id = ref.reference.getId()
      return med as Medication
    )
  )

define fluent function preferredMedCode(med Medication):
  med.code.preferredMedicationCode()

define fluent function preferredMedicationCode(c FHIR.CodeableConcept):
  Coalesce((c.coding) c3 return c3.preferredMedicationCoding())

define fluent function preferredMedicationCoding(c FHIR.Coding):
  case
    when c.system.value = 'http://www.nlm.nih.gov/research/umls/rxnorm' then
      System.Code { system: 'http://www.nlm.nih.gov/research/umls/rxnorm', code: c.code.value, display: c.display.value }
    when c.system.value = 'http://hl7.org/fhir/sid/ndc' then
      System.Code { system: 'http://hl7.org/fhir/sid/ndc', code: c.code.value, display: c.display.value }
    else
      null
  end
