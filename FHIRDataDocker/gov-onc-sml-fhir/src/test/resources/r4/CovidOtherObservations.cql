library CovidOtherObservations version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.1.000'
include FHIRCommon version '1.0.0' called FC
include CovidConcepts called Cx

context Patient

/// Report QA analysis with count of resources, no PHI.
define ReportOtherObservationsQA: 
{
  "Smoking Status Count": Count("Tobacco smoking status"),

  "Brain Fog Count": Count("Brain Fog"),
  "Impaired Sense of Smell Count": Count("Impaired Sense of Smell"),
  "Impaired Sense of Taste Count": Count("Impaired Sense of Taste"),
  "Post Exertional Malaise Count": Count("Post Exertional Malaise"),
  "Respiratory Distress Count": Count("Respiratory Distress"),
  "Social History Count": Count("Social History Observations")
}

/// Report a list of codes for all other observations.
define ReportOtherObservations: 
{
  "Smoking Status": "Tobacco smoking status".mostRecent().resultValue(),

  "Brain Fog": "Brain Fog".mostRecent().resultValue(),
  "Impaired Sense of Smell": "Impaired Sense of Smell".mostRecent().resultValue(),
  "Impaired Sense of Taste": "Impaired Sense of Taste".mostRecent().resultValue(),
  "Post Exertional Malaise": "Post Exertional Malaise".mostRecent().resultValue(),
  "Respiratory Distress": "Respiratory Distress".mostRecent().resultValue()
}

define "Social History Observations":
  (["Observation"] O
    where O.isSocialHistory()).resulted()

// define fluent function observationCodes(observations List<Observation>):
//   (observations observation
//     return observation.firstCode().code)
  
// Returns a System.Code with sytem and code parameters for first Observation coding
// define fluent function firstCode(observation Observation):
//   First(
//     observation.code.coding c
//       return System.Code { system: c.system.value, code: c.code.value, display: c.display.value }
//   )

// define fluent function firstValueCode(obs Observation):
//   if obs.value is CodeableConcept then
//     (obs.value as CodeableConcept).conceptCode()
//   else
//     null

// define fluent function conceptCode(c FHIR.CodeableConcept):
//   Coalesce((c.coding) c2 return c2.code.value)

define "Tobacco smoking status":
  ([Observation: Cx."Tobacco smoking status"]).resulted()

define "Brain Fog":
  ([Observation: Cx."Brain Fog"]).resulted()

define "Impaired Sense of Smell":
  ([Observation: Cx."Impaired Sense of Smell"]).resulted()

define "Impaired Sense of Taste":
  ([Observation: Cx."Impaired Sense of Taste"]).resulted()

define "Post Exertional Malaise":
  ([Observation: Cx."Post Exertional Malaise"]).resulted()

define "Respiratory Distress":
  ([Observation: Cx."Respiratory Distress"]).resulted()
