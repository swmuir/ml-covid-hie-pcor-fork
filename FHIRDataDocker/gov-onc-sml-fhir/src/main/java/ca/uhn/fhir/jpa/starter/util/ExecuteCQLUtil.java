package ca.uhn.fhir.jpa.starter.util;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.cqframework.cql.cql2elm.DefaultLibrarySourceProvider;
import org.hl7.elm.r1.VersionedIdentifier;
import org.hl7.fhir.instance.model.api.IBase;
import org.hl7.fhir.instance.model.api.IBaseBundle;
import org.hl7.fhir.instance.model.api.IBaseDatatype;
import org.hl7.fhir.instance.model.api.IBaseResource;
import org.opencds.cqf.cql.engine.execution.EvaluationResult;
import org.opencds.cqf.cql.engine.execution.ExpressionResult;
import org.opencds.cqf.fhir.api.Repository;
import org.opencds.cqf.fhir.cql.CqlOptions;
import org.opencds.cqf.fhir.cql.Engines;
import org.opencds.cqf.fhir.cql.EvaluationSettings;
import org.opencds.cqf.fhir.cql.engine.retrieve.RetrieveSettings;
import org.opencds.cqf.fhir.cql.engine.retrieve.RetrieveSettings.PROFILE_MODE;
import org.opencds.cqf.fhir.cql.engine.retrieve.RetrieveSettings.SEARCH_FILTER_MODE;
import org.opencds.cqf.fhir.cql.engine.retrieve.RetrieveSettings.TERMINOLOGY_FILTER_MODE;
import org.opencds.cqf.fhir.cql.engine.terminology.TerminologySettings;
import org.opencds.cqf.fhir.cql.engine.terminology.TerminologySettings.CODE_LOOKUP_MODE;
import org.opencds.cqf.fhir.cql.engine.terminology.TerminologySettings.VALUESET_EXPANSION_MODE;
import org.opencds.cqf.fhir.cql.engine.terminology.TerminologySettings.VALUESET_MEMBERSHIP_MODE;
import org.opencds.cqf.fhir.cql.engine.terminology.TerminologySettings.VALUESET_PRE_EXPANSION_MODE;
import org.opencds.cqf.fhir.utility.repository.InMemoryFhirRepository;
import org.opencds.cqf.fhir.utility.repository.ProxyRepository;
import org.opencds.cqf.fhir.utility.repository.ig.IgRepository;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.context.FhirVersionEnum;

public class ExecuteCQLUtil {

	private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(ExecuteCQLUtil.class);

	private static String cqlLibraries;

	private static String cqlVocabulary;

	private static String cqlPlan;

	private static String cqlPlanFolder;

	public static String getCQLPlansFolder() {
		return cqlPlanFolder;
	}

	public static void setCQLPlansFolder(String cqlPlanFolder) {
		ExecuteCQLUtil.cqlPlanFolder = cqlPlanFolder;
	}

	public static void setCqlPlan(String cqlPlan) {
		ExecuteCQLUtil.cqlPlan = cqlPlan;
	}

	public static void setCqlLibraries(String cqlLibraries) {
		ExecuteCQLUtil.cqlLibraries = cqlLibraries;
	}

	public static void setCqlVocabulary(String cqlValuesets) {
		ExecuteCQLUtil.cqlVocabulary = cqlValuesets;
	}

	Repository content = null;
	Repository terminology = null;

	public EvaluationResult executeCQL(String libraryName, IBaseBundle bundle) throws Exception {

		FhirVersionEnum fhirVersionEnum = FhirVersionEnum.R4;

		FhirContext fhirContext = FhirContext.forCached(fhirVersionEnum);
		CqlOptions cqlOptions = CqlOptions.defaultOptions();

		var terminologySettings = new TerminologySettings();
		terminologySettings.setValuesetExpansionMode(VALUESET_EXPANSION_MODE.PERFORM_NAIVE_EXPANSION);
		terminologySettings.setValuesetPreExpansionMode(VALUESET_PRE_EXPANSION_MODE.USE_IF_PRESENT);
		terminologySettings.setValuesetMembershipMode(VALUESET_MEMBERSHIP_MODE.USE_EXPANSION);
		terminologySettings.setCodeLookupMode(CODE_LOOKUP_MODE.USE_CODESYSTEM_URL);

		var retrieveSettings = new RetrieveSettings();
		retrieveSettings.setTerminologyParameterMode(TERMINOLOGY_FILTER_MODE.FILTER_IN_MEMORY);
		retrieveSettings.setSearchParameterMode(SEARCH_FILTER_MODE.FILTER_IN_MEMORY);
		retrieveSettings.setProfileMode(PROFILE_MODE.DECLARED);

		var evaluationSettings = EvaluationSettings.getDefault();
		evaluationSettings.setCqlOptions(cqlOptions);
		evaluationSettings.setTerminologySettings(terminologySettings);
		evaluationSettings.setRetrieveSettings(retrieveSettings);

		Repository repository = createRepository(fhirContext, bundle, null);

		var engine = Engines.forRepositoryAndSettings(evaluationSettings, repository, null, null, true);

		var provider = new DefaultLibrarySourceProvider(Path.of(cqlLibraries));
		engine.getEnvironment().getLibraryManager().getLibrarySourceLoader().registerProvider(provider);

		VersionedIdentifier identifier = new VersionedIdentifier().withId(libraryName);

		Pair<String, Object> contextParameter = null;
		EvaluationResult result = engine.evaluate(identifier, contextParameter);

		writeResult(result);

		return result;
	}

	private Repository createRepository(FhirContext fhirContext, IBaseBundle bundle, String contextValue) {

		if (content == null) {
			Path path = Path.of(cqlLibraries);
			content = new IgRepository(fhirContext, path);
		}
		Repository data = new InMemoryFhirRepository(fhirContext, bundle);
		terminology = new IgRepository(fhirContext, Paths.get(cqlVocabulary));
		return new ProxyRepository(data, content, terminology);
	}

	@SuppressWarnings("java:S106") // We are intending to output to the console here as a CLI tool
	private void writeResult(EvaluationResult result) {
		for (Map.Entry<String, ExpressionResult> libraryEntry : result.expressionResults.entrySet()) {
			System.out.println(libraryEntry.getKey() + "=" + this.tempConvert(libraryEntry.getValue().value()));
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
			result = resource.fhirType() + (resource.getIdElement() != null && resource.getIdElement().hasIdPart()
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

}
