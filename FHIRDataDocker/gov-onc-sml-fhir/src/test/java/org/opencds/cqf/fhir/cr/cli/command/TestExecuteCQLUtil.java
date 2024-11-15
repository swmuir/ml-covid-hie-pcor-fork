package org.opencds.cqf.fhir.cr.cli.command;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.jena.atlas.json.JSON;
import org.apache.jena.atlas.json.JsonObject;
import org.apache.jena.atlas.json.JsonValue;
import org.hl7.fhir.instance.model.api.IBase;
import org.hl7.fhir.instance.model.api.IBaseBundle;
import org.hl7.fhir.instance.model.api.IBaseDatatype;
import org.hl7.fhir.instance.model.api.IBaseResource;
import org.hl7.fhir.r4.model.Bundle;
import org.hl7.fhir.r4.model.Encounter;
import org.hl7.fhir.r4.model.Observation;
import org.hl7.fhir.r4.model.Reference;
import org.junit.jupiter.api.Test;
import org.opencds.cqf.cql.engine.execution.EvaluationResult;
import org.opencds.cqf.cql.engine.execution.ExpressionResult;
import org.opencds.cqf.cql.engine.runtime.Tuple;

import ca.uhn.fhir.context.ConfigurationException;
import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.jpa.starter.common.ApplyCQL.DataPoint;
import ca.uhn.fhir.jpa.starter.common.ApplyCQL.DataPointMapper;
import ca.uhn.fhir.jpa.starter.util.ExecuteCQLUtil;
import ca.uhn.fhir.parser.DataFormatException;
import ca.uhn.fhir.util.BundleBuilder;

class TestExecuteCQLUtil {

	@Test
	void test() throws Exception {
		
		
		ExecuteCQLUtil executeCQLUtil = new ExecuteCQLUtil();
		
		String terminology = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/vocabulary";
		
		String library = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/cql";
		
      	String bundlePath = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/src/test/resources/synthea1/Scot349_Pfeffer420_8de9b192-4cdf-51fe-7543-d4ea4ba4f4ac.json";
    	// Parse it
//    	Bundle bundle = parser.parseResource(Bundle.class, "asdf");
    	Bundle bundle = null;
		try {
			bundle = (Bundle) FhirContext.forR4Cached().newJsonParser().parseResource(Files.readString(Path.of(bundlePath)));
		} catch (ConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (DataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		ExecuteCQLUtil.setCqlVocabulary(terminology);
		ExecuteCQLUtil.setCqlLibraries(library);
	
		
	 
			
		
		writeResult(executeCQLUtil.executeCQL( "CovidConditions", bundle));
		writeResult(executeCQLUtil.executeCQL("CovidEncounters", bundle));
		writeResult(executeCQLUtil.executeCQL("CovidVitalSigns", bundle));
		writeResult(executeCQLUtil.executeCQL( "CovidLabResults", bundle));
		writeResult(executeCQLUtil.executeCQL("CovidOtherObservations", bundle));
		writeResult(executeCQLUtil.executeCQL("CovidProcedures", bundle));
	 
		
		
		

	}
	
	@Test
	void test2() throws Exception {
	 
		
		String terminology = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/vocabulary";
		
		String library = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/cql";
		
      	String bundlePath = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/src/test/resources/synthea1/Scot349_Pfeffer420_8de9b192-4cdf-51fe-7543-d4ea4ba4f4ac.json";
    	// Parse it
//    	Bundle bundle = parser.parseResource(Bundle.class, "asdf");
    	Bundle bundle = null;
	 
			bundle = (Bundle) FhirContext.forR4Cached().newJsonParser().parseResource(Files.readString(Path.of(bundlePath)));
		 
		
		ExecuteCQLUtil.setCqlVocabulary(terminology);
		ExecuteCQLUtil.setCqlLibraries(library);
		

		BundleBuilder resultBundleBuilder = new BundleBuilder(FhirContext.forR4Cached());
		
		
		 evaluateCQLPlan(  "cqlplan",bundle,resultBundleBuilder);
		 
		System.err.println( FhirContext.forR4Cached().newJsonParser().setPrettyPrint(true).encodeResourceToString( resultBundleBuilder.getBundle()));
		
	
	}
	
	   private void writeResult(EvaluationResult result) {
	        for (Map.Entry<String, ExpressionResult> libraryEntry : result.expressionResults.entrySet()) {
	            System.out.println(libraryEntry.getKey() + "="
	                    + this.tempConvert(libraryEntry.getValue().value()));
	        }

	        System.out.println();
	    }

	    private String tempConvert(Object value) {
	        if (value == null) {
	            return "null";
	        }

	        String result = "";
	        if (value instanceof Iterable) {
	            result += "[";
	            Iterable<?> values = (Iterable<?>) value;
	            for (Object o : values) {
	                result += (tempConvert(o) + ", ");
	            }

	            if (result.length() > 1) {
	                result = result.substring(0, result.length() - 2);
	            }

	            result += "]";
	        } else if (value instanceof IBaseResource) {
	            IBaseResource resource = (IBaseResource) value;
	            result = resource.fhirType()
	                    + (resource.getIdElement() != null
	                                    && resource.getIdElement().hasIdPart()
	                            ? "(id=" + resource.getIdElement().getIdPart() + ")"
	                            : "");
	        } else if (value instanceof IBase) {
	            result = ((IBase) value).fhirType();
	        } else if (value instanceof IBaseDatatype) {
	            result = ((IBaseDatatype) value).fhirType();
	        } else {
	            result = value.toString();
	        }

	        return result;
	    }

//	    public  EvaluationResult  runCQLs(IBaseBundle bundle,String library) {
//			EvaluationResult results = null;
//			try {
//				results = executeCQLUtil.executeCQL(library,bundle);
//			} catch (Exception e) {
//				e.printStackTrace();
//			}
//			return results;
//		}

	    
		public  void  evaluateCQLPlan( String cqlPlan, IBaseBundle bundle,BundleBuilder resultBundle) throws Exception {
			 
			ExecuteCQLUtil executeCQLUtil = new ExecuteCQLUtil();
			
			String terminology = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/vocabulary";
			
			String library = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/cql";
			
			String cqlPlansFolder = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/input/plans/";
			
	      	String bundlePath = "/Users/seanmuir/git/mlrelease/gov-onc-sml-fhir/src/test/resources/synthea1/Scot349_Pfeffer420_8de9b192-4cdf-51fe-7543-d4ea4ba4f4ac.json";
	    	// Parse it
//	    	Bundle bundle = parser.parseResource(Bundle.class, "asdf");
//	    	Bundle bundle = null;
//			try {
//				bundle = (Bundle) FhirContext.forR4Cached().newJsonParser().parseResource(Files.readString(Path.of(bundlePath)));
			ExecuteCQLUtil.setCqlVocabulary(terminology);
			ExecuteCQLUtil.setCqlLibraries(library);
			ExecuteCQLUtil.setCQLPlansFolder(cqlPlansFolder);
		
			
//			logger.trace("evaluateCQLPlan " + cqlPlan + " " + patient);
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
					
					EvaluationResult results = executeCQLUtil.executeCQL( plan.getAsObject().getString("library"), bundle);
					
					writeResult(results);
					

					for (DataPoint dp : dps) {
						System.err.println(dp.key);
//						
//						
//						
						if (results.expressionResults.containsKey(dp.key)) {
							Tuple tuple =	(Tuple) results.expressionResults.get(dp.key).value();
							System.err.println(tuple.toString());
							for (DataPoint dp2 :dp.dataPoints) {
//								
								System.err.println(dp2.key + " " + dp2.code);
//								
//								
//									
									Observation observation = new Observation();
									
									observation.getCode().addCoding().setCode(dp2.code ).setDisplay(dp2.key);
//									
									resultBundle.addCollectionEntry(observation);
									String valueKey = dp2.key.replaceAll(",", " ");
//									
//									observation.setSubject(new Reference(thePatient));
									if (tuple.getElements().containsKey(dp2.key) && tuple.getElement(dp2.key) != null) {
										
										System.err.println(dp2.key + " " + dp2.code + " === " + tuple.getElement(dp2.key));
										
//										logger.trace(dp2.key + " " + dp2.code + " === " + tuple.getElement(dp2.key));
										String normalized = (String) tuple.getElement(dp2.key).toString();
										
										System.err.println(dp2.key + " " + dp2.code + " (normalized) === " + normalized);
										
										observation.getValueStringType().setValue(normalized);
										
//										if (cqlCache != null) {
//											cqlCache.put(dp2.code, (String) tuple.getElement(dp2.key).toString());
//										}
//
//										observation.getValueStringType()
//												.setValue(annotationContext
//														? thePatient.getIdPart() + " "
//																+ (encounter != null ? encounter.getIdPart()
//																		: "NO ENCOUNTER")
//																+ " " + valueKey + "  "
//																+ (String) tuple.getElement(dp2.key).toString()
//														: normalized);
									} else {
										
										observation.getValueStringType().setValue("NoData");
										
////									System.err.println("evaluateCQLPlan " + cqlPlan + " " + patient + " item  " + dp2.key + " NO RESULT ");
////									logger.trace(dp2.key + " " + dp2.code + " ===   NO RESULT ");
//
//									observation.getValueStringType()
//											.setValue(
//													annotationContext
//															? thePatient.getIdPart() + " "
//																	+ (encounter != null ? encounter.getIdPart()
//																			: "NO ENCOUNTER")
//																	+ " " + valueKey + "  " + "NULL"
//															: ( continual && cqlCache != null &&  cqlCache.containsKey(dp2.code) ? cqlCache.get(dp2.code)
//											}						: ""));
								}
								
							}
						}

					}
				}
//			}
		}
		}


