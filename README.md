# ml-covid-hie-pcor
This project intends to create a foundation to use electronic health data from HIEs for PCOR by implementing data standards, APIs, and privacy-preserving machine learning (ML) infrastructure.

# ONC-SplitLearning
APPLICATIONS OF MACHINE LEARNING TO HEALTH DATA FOR PATIENT-CENTERED COVID-19 RESEARCH
BookZurman Inc
C.1	PROGRAM DESCRIPTION AND BACKGROUND

The COVID-19 pandemic has created an urgent need to expand data access for
researchers, healthcare providers, and policy-makers. Most electronic health data are highly decentralized, managed across many different provider institutions, and not easily accessible to researchers. Similarly, data acquisition problems have included incomplete data, lag
of timely data, and data that are not paired with demographic information such as race and ethnicity.

State and regional Health Information Exchanges (HIEs) collect patient-level health information as part of their activities for care coordination, including from more than 60% of US
hospitals. They are valuable and underutilized resources for driving discovery using large-scale population data. HIEs cover patient populations of varying size over a specific local or state geography, receiving data from a high density of healthcare providers in their coverage
area. State and local HIEs are designed to manage the health of a population by exchanging patient health information with clinicians, public health agencies, and laboratories and linking, analyzing, and aggregating that data. The application of artificial intelligence (AI) and machine learning (ML) techniques to HIE data can tap a new data source for conducting complex patient outcomes research studies and further understanding COVID-19 and its progression.

Despite the availability of robust patient-level electronic health data in HIEs, their datasets are rarely used for research purposes due to technical and privacy related barriers. This project proposes to make advances to address two significant reasons for the inaccessibility of HIE data to researchers.

First, lack of data standardization across HIEs can result in data that is less useful for secondary uses such as research. It also affects the ability to apply advanced technologies such as AI and ML. This will be addressed by implementing the United States Core Data for Interoperability (USCDI) and the Bulk Fast Healthcare Interoperability Resources (FHIR) API. USCDI is a standardized set of health data classes and constituent data elements for nationwide, interoperable health information exchange. The Bulk FHIR API supports scaling and efficient access to large amounts of data for a population of individuals.

Second, privacy issues inhibit data sharing across HIEs and the use of their data for research. This project will test the application of split learning, a privacy-preserving ML technique. Split learning is seen as a novel solution to overcome privacy concerns by removing the need to share raw data. Traditionally, application of ML requires centralization of large amounts of data to create accurate models. While sharing raw data among HIEs would be ideal, this is challenging due to privacy and security regulations. Instead, split learning allows each entity to generate partial insights, which are then layered into a single ML model.
 


C.2	STATEMENT OF WORK – Separate Nonseverable Services Components

The Contractor shall furnish all of the necessary personnel, materials, services, facilities, (except as otherwise specified herein), and otherwise do all the things necessary for or incident to the performance of the work as set forth below:

C.2.1	SERVICES COMPONENT 1

Purpose:

The purpose of this contract is to engage with three HIEs to develop an ML model that addresses research question(s) related to COVID-19 and patient-centered outcomes. This will result in a comprehensive, public final report and associated deliverables that detail the steps taken, lessons learned, best practices, and other information that can assist future use of split learning and the use of HIE data for research.

This project contributes to the Department of Health and Human Services strategic goal of strengthening and modernizing the nation’s data infrastructure by creating a foundation to use electronic health data from HIEs for patient-centered outcomes research (PCOR).

End Product of Value:

The final product of this contract shall be an ML model that addresses research question(s) related to COVID-19 and patient-centered outcomes. The model shall apply the split learning technique across select data from three HIEs.

For the purposes of this contract, a machine learning algorithm refers to the procedure that is run on data in order to create the machine learning model, which is the output to be run on new data to make inferences.

The contractor shall be responsible for:

1	OBJECTIVE TASKS

Task 1.1 HIE Selection

1.1.A	HIE Selection
Based on an environmental scan previously conducted by ONC, the COR will provide to the Contractor a prioritized list of HIEs that have the technical capacity and experience to participate in the project. The Contractor shall engage with these HIEs on their interest in participating in the project and continuing its work following project completion, culminating in formalized agreements with three HIEs.
 

Task 1.2 HIE Infrastructure Preparation

1.2.A	Prepare Infrastructure of HIEs
The Contractor shall engage with the HIEs to prepare the HIEs’ infrastructure for developing and implementing a machine learning model using split learning. This shall include implementing USCDI and Bulk FHIR API if those specifications are not already implemented.

The Contractor shall take all necessary steps to ensure that all patient personally identifying information (PII) and protected health information (PHI) remain secure.

As sub-milestones specified in the Work Plan are completed, the Contractor shall demonstrate them to the COR for verification and approval.

1.2.B	Development Tracker
As modification steps are taken, the Contractor shall maintain a Development Tracker logging information including dates, issue descriptions, resolutions, and lessons learned.

After completion of 1.2.A tasks, the Contractor shall present a summary presentation of Development Tracker items.

1.2.C	Midpoint Summary Report
Once infrastructure modifications have been completed, the Contractor shall draft a summary report of early project activities and findings to be made available on ONC’s website (www.healthit.gov) and disseminated to target audiences.

The Contractor shall submit the report to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

Task 1.3 Select Research Questions and Develop Split Learning Model

1.3	.A Select Research Questions
With guidance from the COR, the Contractor shall propose patient-centered outcomes research questions and related ML algorithms and models regarding COVID-19 that will be applied to the relevant data available across the HIEs. The research questions shall involve topics such as socially determinant risk factors, vulnerable population factors, behavioral/mental health
factors, racial equity, factors unique to people with chronic conditions, and other related topics. At the conclusion of a process winnowing down the proposed questions, which may
include proposals sourced from other entities, the COR will select up to three research questions to investigate.

Once the research questions have been selected by the COR, the Contractor shall prepare a document describing each question including:
•	The question’s background context;
•	Value to researchers, clinicians, PCOR, and others;
 

•	Why the question is appropriate for ML/split learning compared to simpler analytical methods;
•	What HIE data would be used to address the question and what algorithms would be used to train the model;
•	A validation and testing approach to evaluate model performance and thresholds for accuracy and performance of the model;
•	Any other considerations, technical and otherwise, related to the execution of the research and model.

The Contractor shall submit the document to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

1.3	.B Develop Split Learning Model
The Contractor shall develop the split learning model that addresses the research question(s) selected by the COR. This shall include, but not be limited to, the following steps:
•	Acquire and implement HIE servers (one each, so three total) and aggregation server;
•	Prepare, transform, and train relevant HIE data within HIE servers to be used for the model;
•	Securely transmit datasets and initial model layers to aggregation server;
•	Apply algorithms to aggregated dataset and model layers from HIEs;
•	Assess results and continue to train model until results meet quality thresholds.

The Contractor shall take all necessary steps to ensure that all patient personally identifying information (PII) and protected health information (PHI) remain secure.

As sub-milestones specified in the Work Plan are completed, the Contractor shall demonstrate them to the COR for verification and approval.

1.3.C Deploy Split Learning Model
When results meet quality thresholds, the Contractor shall deploy the final model back to each HIE and apply the final model to HIE datasets for inference.

The Contractor shall take all necessary steps to ensure that all patient personally identifying information (PII) and protected health information (PHI) remain secure.

As sub-milestones specified in the Work Plan are completed, the Contractor shall demonstrate them to the COR for verification and approval.

1.3.D Development Tracker
As development steps are taken, the Contractor shall continue to maintain a Development Tracker logging information including dates, issue descriptions, resolutions, and lessons learned.
 

After completion of 1.3.C tasks, the Contractor shall present a summary presentation of Development Tracker items.

1.3.E Split Learning Model Draft Report
The Contractor shall prepare a draft report describing the results of the final model. The report shall include, but not be limited to, how the model was developed, the implications of the model results, and how the model changed as the process proceeded. The report shall serve as the initial draft of the project Final Report.

The Contractor shall submit the report to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

Task 1.4 Communicate and Disseminate Resources

1.4.A Dissemination Plan
The Contractor shall prepare a dissemination plan that identifies opportunities for publication and presentation of the project’s activities and results. The plan shall include, but not be limited to, the Task 1.4 deliverables and actions listed below.

The Contractor shall submit the plan to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

All deliverables and materials that are to be made available to the public in any way shall be made 508 compliant as a requirement for release.

1.4.B Project Home Page
The project home page will branch off the “Building Data Infrastructure to Support Patient- Centered Outcomes Research (PCOR)” page located on the ONC website and will be the primary source housing information and links about the project, activities, and deliverables. The home page shall be active no later than four months after project kickoff.

The Contractor shall prepare the home page content. The Contractor shall submit the content to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR. The Contractor shall work with the COR on its posting to the home page.

1.4.C Implementation Guide
As the model is developed, the Contractor shall prepare an implementation guide (IG) (sample IG can be found here). The IG will include, but not be limited to, describing and detailing the development and implementation of the split learning model in a multi-HIE. The IG will
be hosted on ONC’s website and/or link to an appropriate website, such as ONC’s GitHub page (https://github.com/onc-healthit).

The Contractor shall submit the IG to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.
 


1.4	.D Open Source Code
The Contractor shall create open source code (sample package can be found here) for at least one split learning model, to be hosted on ONC’s website and/or link to an appropriate website, such as ONC’s GitHub page.

The Contractor shall submit the code to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

1.4	.E Journal Manuscript
The Contractor shall write a manuscript for submission to at least one major journal. The target audience will be researchers investigating COVID-19 related patient outcomes and how results of this work advance PCOR. The Contractor shall assess the journal options, submit the manuscript to the journal selected by the COR, and manage the journal process through the manuscript’s publication.

The Contractor shall submit the manuscript to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

1.4.F Public Project Results Presentation
At least one public presentation communicating project results, to be conducted by the Contractor.

The Contractor shall submit the slides to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

Task 1.5 Final Report

The Contractor shall write a final report, to be made publicly available, that will include, but not be limited to, detailed descriptions of the project activities, methods, and lessons learned from the full project and the tasks undertaken. The report will build off of previous deliverables, particularly the mid-point summary report of early project activities, the IG, the preliminary model report, and the development tracker. It will include strengths and limitations of the application of split learning for PCOR and outline next steps to improve the availability and use of HIE data to advance PCOR and pandemic response efforts. The Contractor shall write an associated blog post.

The Contractor shall submit the report to the COR for review and shall address the COR’s verbal and written feedback until approved by the COR.

The final report shall serve as the final deliverable to be delivered to the COR at the conclusion of this task order, representing completion of the contract.

