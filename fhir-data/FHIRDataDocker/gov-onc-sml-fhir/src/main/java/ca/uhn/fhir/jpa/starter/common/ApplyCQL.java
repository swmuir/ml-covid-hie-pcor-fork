package ca.uhn.fhir.jpa.starter.common;

import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.jena.atlas.json.JsonValue;
import org.hl7.fhir.instance.model.api.IBaseBundle;
import org.hl7.fhir.instance.model.api.IBaseResource;
import org.hl7.fhir.r4.model.IntegerType;
import org.hl7.fhir.r4.model.Patient;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.jpa.api.dao.IFhirResourceDaoPatient;
import ca.uhn.fhir.jpa.api.dao.PatientEverythingParameters;
import ca.uhn.fhir.jpa.searchparam.SearchParameterMap;
import ca.uhn.fhir.rest.annotation.Operation;
import ca.uhn.fhir.rest.api.server.IBundleProvider;
import ca.uhn.fhir.rest.param.TokenOrListParam;
import ca.uhn.fhir.rest.param.TokenParam;
import ca.uhn.fhir.rest.server.IResourceProvider;
import ca.uhn.fhir.rest.server.RestfulServer;
import ca.uhn.fhir.util.BundleBuilder;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

public class ApplyCQL {

	RestfulServer fhirServer;

	public ApplyCQL(RestfulServer fhirServer) {
		this.fhirServer = fhirServer;

	}

	@Operation(name = "$applycql", idempotent = true)
	public IBaseBundle applyCQL(HttpServletRequest theServletRequest,
			HttpServletResponse theServletResponse,
			ca.uhn.fhir.rest.api.server.RequestDetails theRequestDetails) {
		BundleBuilder root = new BundleBuilder(this.fhirServer.getFhirContext());

		ca.uhn.fhir.jpa.rp.r4.PatientResourceProvider prp = null;
		ca.uhn.fhir.jpa.rp.r4.ObservationResourceProvider orp = null;

		for (IResourceProvider resourceProvider : fhirServer.getResourceProviders()) {
			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.PatientResourceProvider) {
				prp = (ca.uhn.fhir.jpa.rp.r4.PatientResourceProvider) resourceProvider;
			}

			if (resourceProvider instanceof ca.uhn.fhir.jpa.rp.r4.ObservationResourceProvider) {
				orp = (ca.uhn.fhir.jpa.rp.r4.ObservationResourceProvider) resourceProvider;
			}

		}

		SearchParameterMap searchMap = new SearchParameterMap();
		;
		searchMap.setCount(10000);
		IBundleProvider pb = prp.getDao().search(searchMap, theRequestDetails);

		BundleBuilder observationsBundle = new BundleBuilder(this.fhirServer.getFhirContext());

		for (IBaseResource p : pb.getAllResources()) {
			Patient theTargetPatient = (Patient) p;
			BundleBuilder patientBundle = new BundleBuilder(this.fhirServer.getFhirContext());

			patientBundle.addCollectionEntry(p);
			IntegerType theCount = new IntegerType();
			theCount.setValue(1000);

			PatientEverythingParameters everythingParams = new PatientEverythingParameters();
			everythingParams.setCount(theCount);

			TokenOrListParam retVal = new TokenOrListParam();
			retVal.addOr(new TokenParam(theTargetPatient.getIdElement().getIdPart()));
			IBundleProvider everythings2 = ((IFhirResourceDaoPatient<?>) prp.getDao())
					.patientTypeEverything(theServletRequest, theRequestDetails, everythingParams, retVal);

			for (IBaseResource resource : everythings2.getResources(0, 1000)) {
				root.addCollectionEntry(resource);
			}

			int ctr = 1;
//			LocalRetrieveProvider lpr = new LocalRetrieveProvider(FhirContext.forR4(), root.getBundle());

//			for (JsonObject plans : ExecuteCQLUtil.getPlans()) {
//
//				List<JsonValue> foo = plans.getArray("plan").collect(Collectors.toList());
//
//				for (JsonValue plan : plans.getArray("plan").collect(Collectors.toList())) {
//
//					Function<JsonValue, DataPoint> mymapper = new Function<JsonValue, DataPoint>() {
//						@Override
//						public DataPoint apply(JsonValue t) {
//							DataPoint dp = new DataPoint();
//							dp.key = t.getAsObject().getString("key");
//							dp.code = t.getAsObject().getString("code");
//							dp.system = t.getAsObject().getString("system");
//							return dp;
//						}
//					};
//
//					HashSet<DataPoint> datapoints = new HashSet<>();
//
//					datapoints.addAll(
//							plan.getAsObject().getArray("datapoints").map(mymapper).collect(Collectors.toSet()));
//
//					Set<String> datapoints2 = new HashSet<>();
//					datapoints2.addAll(datapoints.stream().map(DataPoint::getKey).collect(Collectors.toSet()));
//
////					RetrieveProvider lpr;
//					HashMap<String, Object> results = ExecuteCQLUtil.executeCQL("example",
//							plan.getAsObject().getString("library"), "R4", lpr, datapoints2);
//
////					HashMap<String, Object> results = ExecuteCQLUtil.executeCQL(thePatient, library, "R4", bundleProvider,
////							datapoints);
//
//					for (DataPoint dp : datapoints) {
//
//						IFhirResourceDao<Observation> odao = orp.getDao();
//						Observation observation = new Observation();
//						RequestDetails request = new ServletRequestDetails();
//						;
//
//						observation.setSubject(new Reference(theTargetPatient.getId()));
//
//						CodeableConcept cc = new CodeableConcept();
//
//						cc.setText(dp.key);
//
//						cc.addCoding().setCode(dp.code).setDisplay(dp.key).setSystem(dp.system);
//
//						observation.setCode(cc);
//
//						StringType value = new StringType();
//
//						value.setValue((String) results.get(dp.key));
//
//						observation.setValue(value);
//
//						DaoMethodOutcome result = odao.create(observation, request);
//
//						IParser jp = this.fhirServer.getFhirContext().newJsonParser();
//						jp.setPrettyPrint(true);
//
//						observationsBundle.addCollectionEntry(result.getResource());
//
//					}
//				}
//			}
		}

		return observationsBundle.getBundle();
	}

	public static class DataPoint {
		public String key;

		public String getKey() {
			return key;
		}

		public String getCode() {
			return code;
		}

		public String getSystem() {
			return system;
		}

		public String code;
		public String system;

		public Set<DataPoint> dataPoints;
	}

	public static class DataPointMapper implements Function<JsonValue, DataPoint> {
		public DataPoint apply(JsonValue t) {
			DataPoint dp = new DataPoint();

			if (t.getAsObject().hasKey("key")) {
				dp.key = t.getAsObject().getString("key");
			}
			if (t.getAsObject().hasKey("code")) {
				dp.code = t.getAsObject().getString("code");
			}
			if (t.getAsObject().hasKey("system")) {
				dp.system = t.getAsObject().getString("system");
			}

			if (t.getAsObject().hasKey("datapoints")) {
				dp.dataPoints = t.getAsObject().getArray("datapoints").map(this).collect(Collectors.toSet());
			}

			return dp;
		}
	}

}
