package org.onc.ml.transformation.tests;
/*******************************************************************************
 * Copyright (c) 2023 seanmuir.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     seanmuir - initial API and implementation
 *
 *******************************************************************************/


import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.io.FilenameUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mdmi.rt.service.web.Application;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.SpringBootTest.WebEnvironment;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

/**
 * @author seanmuir
 *
 */

@RunWith(SpringRunner.class)
@SpringBootTest(classes = Application.class, webEnvironment = WebEnvironment.RANDOM_PORT)
class OncMLTests {

	@Autowired
	private TestRestTemplate template;

	@BeforeAll
	public static void setEnvironment() {
		System.setProperty("mdmi.maps", "src/main/resources/maps");
		
		System.setProperty("MDMI_SOURCE_FILTER_FLAG","TRUE");
		
		System.setProperty("LOGGING_LEVEL_MDMI","INFO");
	}

	static int increment=1;
	private String runTransformation(String source, String target, String message,String extension,String testPathExt) throws Exception {
		MultiValueMap<String, Object> map = new LinkedMultiValueMap<>();
		map.add("source", source);
		map.add("target", target);
		map.add("message", new FileSystemResource(Paths.get(message)));
		ResponseEntity<String> response = template.postForEntity("/mdmi/transformation", map, String.class);
		System.out.println(response.getStatusCode());
		assertTrue(response.getStatusCode().equals(HttpStatus.OK));
		Path sourcePath = Paths.get(message);
		String testName = FilenameUtils.removeExtension(sourcePath.getFileName().toString()) ;

		Path testPath = Paths.get("target/test-output/" +testPathExt +"/"+ target + testName);
		if (!Files.exists(testPath)) {
			Files.createDirectories(testPath);
		}

		Path path = Paths.get("target/test-output/" +testPathExt +"/" + target + testName + "/" + testName + increment++ + "." +extension);
		byte[] strToBytes = response.getBody().getBytes();

		Files.write(path, strToBytes);

		// System.out.println(response.getBody());
		return response.getBody();
	}

	@Test
	public void testSyntheaFHIR2M2L() throws Exception {
		
		Set<Path> documents3 = Files.walk(Paths.get("src/test/resources/samples/bundleofbundles")).filter(Files::isRegularFile) .collect(Collectors.toSet()); 
		for (Path document: documents3) {					
				runTransformation("FHIRR4JSON.MasterBundle", "ONCML.PHASE2", document.toAbsolutePath().toString(),"csv","bundleofbundles");			 
		}
	}
	
	@Test
	public void testSyntheaFHIR2M2LA() throws Exception {
		
		Set<Path> documents3 = Files.walk(Paths.get("src/test/resources/samples/bundleofbundlesA")).filter(Files::isRegularFile) .collect(Collectors.toSet()); 
		for (Path document: documents3) {	
			for (int ii=0; ii < 500; ii++) {
				runTransformation("FHIRR4JSON.MasterBundle", "ONCML.PHASE2", document.toAbsolutePath().toString(),"csv","bundleofbundlesA");
			}
		}
	}

	@Test
	public void testSyntheaFHIR2MLANALYSIS() throws Exception {
		
		Set<Path> documents3 = Files.walk(Paths.get("src/test/resources/samples/bundleofbundlesqa")).filter(Files::isRegularFile) .collect(Collectors.toSet()); 
		for (Path document: documents3) {					
				runTransformation("FHIRR4JSON.MasterBundle", "ONCML.ANALYSIS", document.toAbsolutePath().toString(),"csv","bundleofbundlesqa");	
				break;
		}
	}
	
	@Test
	public void testSyntheaFHIR2MLPATIENTANALYSIS() throws Exception {
		
		Set<Path> documents3 = Files.walk(Paths.get("src/test/resources/samples/bundleofbundlespatientqa")).filter(Files::isRegularFile) .collect(Collectors.toSet()); 
		for (Path document: documents3) {					
				runTransformation("FHIRR4JSON.MasterBundle", "ONCML.ANALYSIS", document.toAbsolutePath().toString(),"csv","patient");	
				break;
		}
	}

	
	@Test
	public void testDropColumn() throws Exception {
		
		Set<Path> documents3 = Files.walk(Paths.get("src/test/resources/samples/3938")).filter(Files::isRegularFile) .collect(Collectors.toSet()); 
		for (Path document: documents3) {		
			for (int ii=0; ii < 5; ii++) {
				System.out.println(ii);
				runTransformation("FHIRR4JSON.MasterBundle", "ONCML.PHASE2", document.toAbsolutePath().toString(),"csv","bundleofbundles");
			}
		}
	}


}
