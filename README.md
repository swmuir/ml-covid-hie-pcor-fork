# Project Background
State and local Health Information Exchanges (HIEs) receive data from a high density of healthcare providers in their coverage area, and, in the aggregate, from more than 60% of US hospitals. HIEs exchange patient health information with clinicians, public health agencies, and laboratories and link, analyze, and aggregate that data. However, despite this availability of robust patient-level electronic health data, these datasets are rarely used for research purposes because of technical and privacy related barriers. Privacy-preserving artificial intelligence (AI) and machine learning (ML) techniques can leverage HIE data to conduct complex patient outcomes research studies and further the understanding of COVID-19 and its progression.

This project contributes to the Department of Health and Human Services strategic goal of strengthening and modernizing the nation’s data infrastructure by creating a foundation to use electronic health data from HIEs for patient-centered outcomes research (PCOR).

# Project Dates
This project began in 2021 and ends in 2024.

# Project Goal
The goal of this project is to create a foundation to use electronic health data from HIEs for PCOR by implementing data standards, APIs, and privacy-preserving machine learning (ML) infrastructure. It will accomplish this by:

* Implementing the United States Core Data for Interoperability (USCDI) and Bulk FHIR API at three HIEs to facilitate interoperable and efficient data access.
* Testing the use of split learning, a privacy-preserving machine learning technique, with HIE data to address a COVID-19 and PCOR-related research question.
* Disseminating resources to support the adoption by HIEs of technologies and methods used in the project and to encourage PCOR researchers to use HIEs and their data for research.

# Components
## fhir-data
This component process FHIR bulk data based on a cohort and transforms to ML data model

## split-learning
This component executes the split learning model on the data from the fhir-data component 
