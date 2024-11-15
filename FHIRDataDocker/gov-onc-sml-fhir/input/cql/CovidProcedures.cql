library CovidProcedures version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.1.000'
include FHIRCommon version '1.0.0' called FC
include CovidConcepts called Cx

context Patient

/// Report QA analysis with count of resources, no PHI.
define ReportProceduresQA: 
{
  "All Procedures Count": Count("All Procedures"),
  "Procedures Without Preferred Code Count": Count("Procedures Without Preferred Code")
}

/// Report a list of codes for all procedures, using preferred coding system.
define ReportProcedures: 
{
  "Procedure Codes": "All Procedures".preferredOrFirstCodes()
}

define "All Procedures":
  [Procedure]

define "Procedures Without Preferred Code":
  "All Procedures" procedure
    where procedure.preferredCode() is null

define fluent function preferredOrFirstCodes(procedures List<Procedure>):
  procedures.preferredCodes()
  union
  (procedures procedure
    where procedure.preferredCode() is  null
    return procedure.firstCode().code)
  
define fluent function preferredCodes(procedures List<Procedure>):
  procedures procedure
    where procedure.preferredCode() is not null
    return procedure.preferredCode().code
  
// Returns a System.Code with sytem and code parameters
define fluent function preferredCode(procedure Procedure):
  First(
    procedure.code.coding c
      let preferredCode: PreferredProcedureCoding(c)
      where preferredCode.code is not null
      return preferredCode
  )

define function PreferredProcedureCoding(c FHIR.Coding):
  case
    when c.system.value = 'http://snomed.info/sct' or c.system.value = 'urn:oid:2.16.840.1.113883.6.96' then
      System.Code { system: 'http://snomed.info/sct', code: c.code.value, display: c.display.value }
    when c.system.value = 'http://www.ama-assn.org/go/cpt' or c.system.value = 'urn:oid:2.16.840.1.113883.6.12' then
      System.Code { system: 'http://www.ama-assn.org/go/cpt', code: c.code.value, display: c.display.value }
    else
      null
  end

// Returns a System.Code with sytem and code parameters for first Procedure coding
define fluent function firstCode(procedure Procedure):
  First(
    procedure.code.coding c
      return System.Code { system: c.system.value, code: c.code.value, display: c.display.value }
  )
