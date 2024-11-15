package ca.uhn.fhir.jpa.starter.common;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.apache.jena.atlas.json.JSON;
import org.apache.jena.atlas.json.JsonObject;
import org.apache.jena.atlas.json.JsonValue;
import org.hl7.fhir.instance.model.api.IBaseBundle;
import org.hl7.fhir.instance.model.api.IBaseResource;
import org.hl7.fhir.instance.model.api.IIdType;
import org.hl7.fhir.r4.model.Bundle;
import org.hl7.fhir.r4.model.Coding;
import org.hl7.fhir.r4.model.Condition;
import org.hl7.fhir.r4.model.DocumentReference;
import org.hl7.fhir.r4.model.DocumentReference.DocumentReferenceContentComponent;
import org.hl7.fhir.r4.model.Encounter;
import org.hl7.fhir.r4.model.IdType;
import org.hl7.fhir.r4.model.Observation;
import org.hl7.fhir.r4.model.Patient;
import org.hl7.fhir.r4.model.Procedure;
import org.hl7.fhir.r4.model.Reference;
import org.hl7.fhir.r4.model.StringType;
import org.opencds.cqf.cql.engine.execution.EvaluationResult;
import org.opencds.cqf.cql.engine.retrieve.RetrieveProvider;
import org.opencds.cqf.cql.engine.runtime.Tuple;

//import com.apicatalog.jsonld.uri.Path;
import com.microsoft.sqlserver.jdbc.StringUtils;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.jpa.rp.r4.ConditionResourceProvider;
import ca.uhn.fhir.jpa.rp.r4.DocumentReferenceResourceProvider;
import ca.uhn.fhir.jpa.rp.r4.EncounterResourceProvider;
import ca.uhn.fhir.jpa.rp.r4.ObservationResourceProvider;
import ca.uhn.fhir.jpa.rp.r4.ProcedureResourceProvider;
import ca.uhn.fhir.jpa.searchparam.SearchParameterMap;
import ca.uhn.fhir.jpa.starter.common.ApplyCQL.DataPoint;
import ca.uhn.fhir.jpa.starter.common.ApplyCQL.DataPointMapper;
import ca.uhn.fhir.jpa.starter.util.ExecuteCQLUtil;
import ca.uhn.fhir.rest.annotation.Operation;
import ca.uhn.fhir.rest.annotation.OperationParam;
import ca.uhn.fhir.rest.api.SortOrderEnum;
import ca.uhn.fhir.rest.api.SortSpec;
import ca.uhn.fhir.rest.api.server.IBundleProvider;
import ca.uhn.fhir.rest.param.ReferenceAndListParam;
import ca.uhn.fhir.rest.param.ReferenceOrListParam;
import ca.uhn.fhir.rest.param.ReferenceParam;
import ca.uhn.fhir.rest.server.IResourceProvider;
import ca.uhn.fhir.rest.server.RestfulServer;
import ca.uhn.fhir.util.BundleBuilder;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

public class PatientEncounterBundle {
	
	private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(PatientEncounterBundle.class);

	
	static private class ObservationKeepByCodePredicate implements Predicate<Observation> {

		List<String> theList;

		/**
		 * @param encounter
		 */
		public ObservationKeepByCodePredicate(List<String> theList) {
			super();
			this.theList = theList;
		}

		@Override
		public boolean test(Observation observation) {
			
			if (observation.getCode() != null) {
				
				for (Coding code : observation.getCode().getCoding() ) {
					
					if ( !StringUtils.isEmpty(code.getCode())) {
					 
						if (theList.contains(code.getCode())) {
							return true;
						}
						
					}
					
				}
			}
			return false;
		}

	}
	
	static private class ObservationPredicate implements Predicate<Observation> {

		Observation theObservation;

		/**
		 * @param encounter
		 */
		public ObservationPredicate(Observation observation) {
			super();
			this.theObservation = observation;
		}

		@Override
		public boolean test(Observation observation) {
			
			if (observation.getCode() != null) {
				
				for (Coding code : observation.getCode().getCoding() ) {
					
					if ( !StringUtils.isEmpty(code.getCode())) {
						
						for (Coding code2 : theObservation.getCode().getCoding()) {
							if (code.getCode().equals(code2.getCode())) {
								return true;
							}
						}
						
					}
					
				}
			}
			return false;
		}

	}

	
	
	static private class AddToBundle implements Consumer<IBaseResource>  {

		BundleBuilder encounterBundle ;
		boolean bundleEmpty = true;
		
		public void setBundleEmpty(boolean bundleEmpty) {
			this.bundleEmpty = bundleEmpty;
		}

		public boolean isBundleEmpty() {
			return bundleEmpty;
		}

		/**
		 * @param encounterBundle
		 */
		public AddToBundle(BundleBuilder encounterBundle) {
			super();
			this.encounterBundle = encounterBundle;
		}

		@Override
		public void accept(IBaseResource t) {
			
			if (bundleEmpty &&  t instanceof Observation) {
				bundleEmpty =false;	
			}
			
			if (bundleEmpty &&  t instanceof Condition) {
				bundleEmpty =false;	
			}
			encounterBundle.addCollectionEntry(t);
			
			
		}
		
	}
	
	
	static private class AddDocumentReferenceToBlob implements Consumer<IBaseResource>  {
 
		StringBuffer blob = new StringBuffer();
		/**
		 * @param encounterBundle
		 */
		public AddDocumentReferenceToBlob() {
			super();
		 
		}

		@Override
		public void accept(IBaseResource t) {
			blob.append(collectDocuments((DocumentReference) t));
		}
		
		public String getBlob() { return blob.toString();}
		
	}
	
	static private class  EncounterConditionPredicate implements Predicate<Condition> {
	
		Encounter encounter;
		String encounterId;
		/**
		 * @param encounter
		 */
		public EncounterConditionPredicate(Encounter encounter) {
			super();
			this.encounter = encounter;
			encounterId = encounter.getId().split("/")[1];
		}

		@Override
		public boolean test(Condition condition) {
               if (condition.getEncounter() != null &&
                    condition.getEncounter().getReference() != null ) {
 				String[] reference = condition.getEncounter().getReference().split("/");
				if (reference[1].equals(encounterId)) {			
					return true;
				}			
			}

			if (condition.hasOnsetDateTimeType() && encounter.hasPeriod()
					&& encounter.getPeriod().getStartElement() != null) {
				if (condition.getOnsetDateTimeType().toCalendar()
						.compareTo(encounter.getPeriod().getStartElement().toCalendar()) <= 0) {
					return true;
				}
			}
			return false;
		}
		
	}
	
	
	static private class  EncounterDocumentReferencePredicate implements Predicate<DocumentReference> {
		
		Encounter encounter;
		String encounterId;
		/**
		 * @param encounter
		 */
		public EncounterDocumentReferencePredicate(Encounter encounter) {
			super();
			this.encounter = encounter;
			encounterId = encounter.getId().split("/")[1];
		}

		@Override
		public boolean test(DocumentReference documentReference) {
			
			if (documentReference.getContext() != null && documentReference.getContext().getEncounter() != null) {
				
				for (Reference ref : documentReference.getContext().getEncounter()) {
					String[] reference = ref.getReference().split("/");
					if (reference[1].equals(encounterId)) {			
					return true;
				}	
				}
				
			}
//			if (condition.getEncounter() != null) {
//				String[] reference = condition.getEncounter().getReference().split("/");
//				if (reference[1].equals(encounterId)) {			
//					return true;
//				}			
//			}
//
//			if (condition.hasOnsetDateTimeType() && encounter.hasPeriod()
//					&& encounter.getPeriod().getStartElement() != null) {
//				if (condition.getOnsetDateTimeType().toCalendar()
//						.compareTo(encounter.getPeriod().getStartElement().toCalendar()) <= 0) {
//					return true;
//				}
//			}
			return false;
		}
		
	}
	
	
	static private class  EncounterObservationPredicate implements Predicate<Observation> {
		
		Encounter encounter;
		String encounterId;

		/**
		 * @param encounter
		 */
		public EncounterObservationPredicate(Encounter encounter) {
			super();
			this.encounter = encounter;			
			encounterId = encounter.getId().split("/")[1];
		}

		@Override
		public boolean test(Observation observation) {
            if (observation.getEncounter() != null && 
                observation.getEncounter().getReference() != null ) {
 			String[] reference = observation.getEncounter().getReference().split("/");
			if (reference[1].equals(encounterId)) {			
				return true;
			}			
		}
		return false;
		} 
		
	}
	
	
	static private class  EncounterProcedurePredicate implements Predicate<Procedure> {
		
		Encounter encounter;
		String encounterId;

		/**
		 * @param encounter
		 */
		public EncounterProcedurePredicate(Encounter encounter) {
			super();
			this.encounter = encounter;
			encounterId = encounter.getId().split("/")[1];
		}

		@Override
		public boolean test(Procedure procedure) {
			
            if (procedure.getEncounter() != null &&
                procedure.getEncounter().getReference() != null ) {
				String[] reference = procedure.getEncounter().getReference().split("/");
				if (reference[1].equals(encounterId)) {			
					return true;
				}			
			}
		
	 

		if (procedure.hasPerformedDateTimeType() && encounter.hasPeriod() && encounter.getPeriod().getStartElement() != null) {
			if (procedure.getPerformedDateTimeType().toCalendar().compareTo(encounter.getPeriod().getStartElement().toCalendar()) <= 0  ) {
				return true;
			}
		}
		return false;
		} 
		
	}
	
	static private class  BinaryOperatorForCondition implements BinaryOperator<Condition> {

		@Override
		public Condition apply(Condition t, Condition u) {
			if (t.hasOnsetDateTimeType() && u.hasOnsetDateTimeType()) {
				if (t.getOnsetDateTimeType().toCalendar().compareTo(u.getOnsetDateTimeType().toCalendar()) <= 0  ) {
					return u;
				}
			}

			return t;
		}
		
	}
	
	static private class  BinaryOperatorForObservation implements BinaryOperator<Observation> {

		@Override
		public Observation apply(Observation t, Observation u) {
			if (t.hasEffectiveDateTimeType() && u.hasEffectiveDateTimeType()) {
				if (t.getEffectiveDateTimeType().toCalendar().compareTo(u.getEffectiveDateTimeType().toCalendar()) <= 0  ) {
					return u;
				}
			}

			return t;
		}
		
	}
	
	static private class  BinaryOperatorForProcedure implements BinaryOperator<Procedure> {

		@Override
		public Procedure apply(Procedure t, Procedure u) {
			if (t.hasPerformedDateTimeType() && u.hasPerformedDateTimeType()) {
				if (t.getPerformedDateTimeType().toCalendar().compareTo(u.getPerformedDateTimeType().toCalendar()) <= 0  ) {
					return u;
				}
			}

			return t;
		}
		
	}
	

	RestfulServer fhirServer;

	public PatientEncounterBundle(RestfulServer fhirServer) {
		this.fhirServer = fhirServer;
		
		executeCQLUtil = new ExecuteCQLUtil();
	}


	private ReferenceAndListParam createParam(String paramValue) {
		ReferenceAndListParam param = new ReferenceAndListParam();
		ReferenceOrListParam param2 = new ReferenceOrListParam();
		ReferenceParam param3 = new ReferenceParam();

		param3.setValue(paramValue);
		param2.add(param3);
		param.addValue(param2);
		return param;

	}

	private static String collectDocuments(DocumentReference theDocuments ) {
		
		StringBuffer buffer = new StringBuffer();
			for (DocumentReferenceContentComponent content :theDocuments.getContent()) {
				
				if (content.getAttachment() != null && content.getAttachment().getData() != null) {		
					buffer.append(content.getAttachment().getDataElement().getValueAsString());					
				}
				
			}
		return buffer.toString();
	}
	
	@Operation(name = "$bundleofbundles", idempotent = true)
	public IBaseBundle bundleofbundles(HttpServletRequest theServletRequest,
			HttpServletResponse theServletResponse,
			ca.uhn.fhir.rest.api.server.RequestDetails theRequestDetails,
			@OperationParam(name = "patient") StringType patientCtr,@OperationParam(name = "plan") StringType cqlPlan,@OperationParam(name = "context") StringType context) throws Exception {

		boolean encounterContext =  true;
		
		boolean annotationContext = false;
		
		boolean cache = true;
		
		if (context != null && context.getValue().equalsIgnoreCase("PATIENT")) {
			encounterContext = false;
			cache = false;
		}
		
		if (context != null && context.getValue().equalsIgnoreCase("ANNOTATION")) {
			annotationContext = true;
			cache = false;
		}
		
		BundleBuilder resultBundleBuilder = new BundleBuilder(FhirContext.forR4Cached());

		ca.uhn.fhir.jpa.rp.r4.PatientResourceProvider prp = null;
		EncounterResourceProvider erp = null;
		ObservationResourceProvider orp = null;
		ConditionResourceProvider crp = null;
		ProcedureResourceProvider procedurerp = null;
		DocumentReferenceResourceProvider drp = null;


		for (IResourceProvider resourceProvider : fhirServer.getResourceProviders()) {
			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.PatientResourceProvider) {
				prp = (ca.uhn.fhir.jpa.rp.r4.PatientResourceProvider) resourceProvider;
			}

			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.EncounterResourceProvider) {
				erp = (ca.uhn.fhir.jpa.rp.r4.EncounterResourceProvider) resourceProvider;
			}

			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.ObservationResourceProvider) {
				orp = (ca.uhn.fhir.jpa.rp.r4.ObservationResourceProvider) resourceProvider;
			}

			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.ConditionResourceProvider) {
				crp = (ca.uhn.fhir.jpa.rp.r4.ConditionResourceProvider) resourceProvider;
			}
			
			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.ProcedureResourceProvider) {
				procedurerp = (ca.uhn.fhir.jpa.rp.r4.ProcedureResourceProvider) resourceProvider;
			}
			
			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.DocumentReferenceResourceProvider) {
				drp = (ca.uhn.fhir.jpa.rp.r4.DocumentReferenceResourceProvider) resourceProvider;
			}

		}

		IIdType pid = new IdType();

		pid.setValue(patientCtr.getValue());

		Patient thePatienta = prp.getDao().read(pid, theRequestDetails);

		org.hl7.fhir.r4.model.Patient thePatient = thePatienta;

		SearchParameterMap paramMap = new SearchParameterMap();

		//paramMap.add("patient", createParam("Patient/" + thePatient.getIdElement().getValue()));
		paramMap.add("subject", createParam("Patient/" + thePatient.getIdElement().getValue()));
		

		List<Condition> theConditions = crp.getDao().search(paramMap, theRequestDetails, theServletResponse).getResources(0, 5000).stream().map(Condition.class::cast).collect(Collectors.toList());

		List<Observation> theObservations = orp.getDao().search(paramMap, theRequestDetails, theServletResponse).getResources(0, 5000).stream().map(Observation.class::cast).collect(Collectors.toList());

		List<Procedure> theProcedures = procedurerp.getDao().search(paramMap, theRequestDetails, theServletResponse).getResources(0, 5000).stream().map(Procedure.class::cast).collect(Collectors.toList());
		
		List<DocumentReference> theDocuments = drp.getDao().search(paramMap, theRequestDetails, theServletResponse).getResources(0, 5000).stream().map(DocumentReference.class::cast).collect(Collectors.toList());

		
		SortSpec thesort = new SortSpec();;
		
		thesort.setParamName("date");
        thesort.setOrder(SortOrderEnum.ASC);
		paramMap.setSort(thesort );

		
		IBundleProvider encounters = erp.getDao().search(paramMap, theRequestDetails, theServletResponse);
		
		
		 
//		int ectr = 0;
//		for (IBaseResource ee :  encounters.getResources(0, 10000)) {
//			Encounter encounter = (Encounter) ee;
//			System.err.println(encounter.getSubject().getDisplay());
//			System.err.println(ee.getIdElement().getValue());
//			System.err.println(ectr++);
//		}
//		
//		int octr = 0;
//		for (Observation oo :  theObservations) {
//			System.err.println(oo.getId());
//			
//			System.err.println(oo.getSubject().getReference());
//			System.err.println(oo.getEncounter().getReference());
//			System.err.println(octr++);
//		}

		
		
		BinaryOperatorForCondition binaryOperatorForCondition = new BinaryOperatorForCondition();		
		BinaryOperatorForObservation binaryOperatorForObservation = new BinaryOperatorForObservation();
		BinaryOperatorForProcedure binaryOperatorForProcedure = new BinaryOperatorForProcedure();
		
		
		List<Observation> observationCarryOvers = new ArrayList<Observation>();
		
		
		if (encounterContext) {
			HashMap<String,String> cqlCache = new HashMap<String,String>();  
			// Assume less the 100 encounters currently
			for (IBaseResource baseResource : encounters.getResources(0, 100)) {
				Encounter encounter = (Encounter) baseResource;
				BundleBuilder encounterBundle = new BundleBuilder(this.fhirServer.getFhirContext());
				BundleBuilder resultBundle = new BundleBuilder(this.fhirServer.getFhirContext());
				AddToBundle addToEncounterBundle = new AddToBundle(encounterBundle);
				EncounterConditionPredicate encounterConditionPredicate = new EncounterConditionPredicate(encounter);
				EncounterObservationPredicate encounteObservationPredicate = new EncounterObservationPredicate(
						encounter);
				EncounterProcedurePredicate encounterProcedurePredicate = new EncounterProcedurePredicate(encounter);
				
				EncounterDocumentReferencePredicate encounterDocumentReferencePredicate = new EncounterDocumentReferencePredicate(encounter);

				encounterBundle.addCollectionEntry(thePatient);
				encounterBundle.addCollectionEntry(encounter);

				resultBundle.addCollectionEntry(thePatient);
				resultBundle.addCollectionEntry(encounter);

				theConditions.stream().filter(encounterConditionPredicate).forEach(addToEncounterBundle);
				theObservations.stream().filter(encounteObservationPredicate).forEach(addToEncounterBundle);
				
				theProcedures.stream().filter(encounterProcedurePredicate).forEach(addToEncounterBundle);

				
				if (!addToEncounterBundle.isBundleEmpty()) {
//					LocalRetrieveProvider lpr = new LocalRetrieveProvider(FhirContext.forR4Cached(),
//							encounterBundle.getBundle());
						evaluateCQLPlan(cqlPlan.asStringValue(), patientCtr.asStringValue(), 	encounterBundle.getBundle(), thePatient,encounter,
								resultBundle,cqlCache, annotationContext);
						
						
						// TODO - Push this logic for adding notes to cqlplan
						AddDocumentReferenceToBlob addDocumentReferenceToResultBundle  = new AddDocumentReferenceToBlob();
						theDocuments.stream().filter(encounterDocumentReferencePredicate).forEach(addDocumentReferenceToResultBundle);
						
						addDocumentReferenceToResultBundle.getBlob();
						
						Observation encounterNotes = new Observation();
						
						encounterNotes.getCode().addCoding().setCode("456db601-2519-42e9-a960-4853c99882d7").setDisplay("Notes");
						
						encounterNotes.getValueStringType().setValue(addDocumentReferenceToResultBundle.getBlob());
						
						resultBundle.addCollectionEntry(encounterNotes);
						
//						Observation encounterTerms = new Observation();
						
//						encounterTerms.getCode().addCoding().setCode("4B4C9AFC-9C0B-4C66-AA6C-00003A78342D").setDisplay("EncounterTerms");
						
//						encounterTerms.getValueStringType().setValue(addDocumentReferenceToResultBundle.getBlob());
						
//						resultBundle.addCollectionEntry(encounterTerms);
						
						

						
						//
						
					resultBundleBuilder.addCollectionEntry(resultBundle.getBundle());
				} else {
					logger.debug("NO OBSERVATIONS FOR " + encounter.getId());
				}

			}
		} else {
			BundleBuilder patientBundle = new BundleBuilder(this.fhirServer.getFhirContext());
			BundleBuilder resultBundle = new BundleBuilder(this.fhirServer.getFhirContext());
			AddToBundle addToPatientBundle = new AddToBundle(patientBundle);
			patientBundle.addCollectionEntry(thePatient);
			resultBundle.addCollectionEntry(thePatient);
			theConditions.stream().forEach(addToPatientBundle);
			theObservations.stream().forEach(addToPatientBundle);

			if (!addToPatientBundle.isBundleEmpty()) {
//				LocalRetrieveProvider lpr = new LocalRetrieveProvider(FhirContext.forR4Cached(),
//						patientBundle.getBundle());
				try {
					evaluateCQLPlan(cqlPlan.asStringValue(), patientCtr.asStringValue(), patientBundle.getBundle(), thePatient,null,
							resultBundle, null,annotationContext);
				} catch (Exception e) {						
					logger.error(e.getLocalizedMessage());
				}
				resultBundleBuilder.addCollectionEntry(resultBundle.getBundle());
			} else {
				logger.debug("NO OBSERVATIONS FOR " + thePatient.getId());
			}
		}

		return resultBundleBuilder.getBundle();
		
		

	}
	
	
	public  EvaluationResult  runCQL(IBaseBundle bundle,String library) {
		EvaluationResult results = null;
		try {
			results = executeCQLUtil.executeCQL(library,bundle);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return results;
	}

	ExecuteCQLUtil executeCQLUtil = null;
  
	 
	public  void  evaluateCQLPlan( String cqlPlan,String patient,IBaseBundle bundle,org.hl7.fhir.r4.model.Patient thePatient,Encounter encounter ,BundleBuilder resultBundle,HashMap<String,String> cqlCache ,boolean  annotationContext) throws Exception {
			 
		
		
		logger.trace("evaluateCQLPlan " + cqlPlan + " " + patient);
		JsonObject	plans = JSON.parse( Files.readString(Path.of( ExecuteCQLUtil.getCQLPlansFolder()   + cqlPlan + ".json")));
		
			for (JsonValue plan : plans.getArray("plan").collect(Collectors.toList())) {

				boolean continual = false;
				if (plan.getAsObject().hasKey("noop")) {
					continue;
				}
				
				if (plan.getAsObject().hasKey("continual")) {
					if ("TRUE".equalsIgnoreCase(plan.getAsObject().getString("continual"))) {
						continual = true;
					}
				}

				Set<DataPoint> dps = plan.getAsObject().getArray("datapoints").map(new DataPointMapper())
						.collect(Collectors.toSet());
				
				EvaluationResult results = runCQL(bundle,plan.getAsObject().getString("library"));
				

				for (DataPoint dp : dps) {
					
					
					
					if (results.expressionResults.containsKey(dp.key)) {
						Tuple tuple =	(Tuple) results.expressionResults.get(dp.key).value();
						System.err.println(tuple.toString());
						for (DataPoint dp2 :dp.dataPoints) {
							
							logger.trace("evaluateCQLPlan " + cqlPlan + " " + patient + " item  " + dp2.key);
							
							
								
								Observation observation = new Observation();
								
								observation.getCode().addCoding().setCode(dp2.code ).setDisplay(dp2.key);
								
								resultBundle.addCollectionEntry(observation);
								String valueKey = dp2.key.replaceAll(",", " ");
								
								observation.setSubject(new Reference(thePatient));
								if (tuple.getElements().containsKey(dp2.key) && tuple.getElement(dp2.key) != null) {
									logger.trace(dp2.key + " " + dp2.code + " === " + tuple.getElement(dp2.key));
									String normalized = (String) tuple.getElement(dp2.key).toString();
									if (cqlCache != null) {
										cqlCache.put(dp2.code, (String) tuple.getElement(dp2.key).toString());
									}

									observation.getValueStringType()
											.setValue(annotationContext
													? thePatient.getIdPart() + " "
															+ (encounter != null ? encounter.getIdPart()
																	: "NO ENCOUNTER")
															+ " " + valueKey + "  "
															+ (String) tuple.getElement(dp2.key).toString()
													: normalized);
								} else {
//								System.err.println("evaluateCQLPlan " + cqlPlan + " " + patient + " item  " + dp2.key + " NO RESULT ");
								logger.trace(dp2.key + " " + dp2.code + " ===   NO RESULT ");

								observation.getValueStringType()
										.setValue(
												annotationContext
														? thePatient.getIdPart() + " "
																+ (encounter != null ? encounter.getIdPart()
																		: "NO ENCOUNTER")
																+ " " + valueKey + "  " + "NULL"
														: ( continual && cqlCache != null &&  cqlCache.containsKey(dp2.code) ? cqlCache.get(dp2.code)
																: ""));
							}
							
						}
					}

				}
			}
//		}

	}

	
	
	

}
