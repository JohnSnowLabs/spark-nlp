---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.0.3
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_0_3
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="prev_ver h3-box" markdown="1">

### 3.0.3

We are glad to announce that Spark NLP for Healthcare 3.0.3 has been released!

#### Highlights

+ Five new entity resolution models to cover UMLS, HPO and LIONC terminologies.
+ New feature for random displacement of dates on deidentification model.
+ Five new pretrained pipelines to map terminologies across each other (from UMLS to ICD10, from RxNorm to MeSH etc.)
+ AnnotationToolReader support for Spark 2.3. The tool that helps model training on Spark-NLP to leverage data annotated using JSL Annotation Tool now has support for Spark 2.3.
+ Updated documentation (Scaladocs) covering more APIs, and examples.

#### Five new resolver models:

+ `sbiobertresolve_umls_major_concepts`: This model returns CUI (concept unique identifier) codes for Clinical Findings, Medical Devices, Anatomical Structures and Injuries & Poisoning terms.            
+ `sbiobertresolve_umls_findings`: This model returns CUI (concept unique identifier) codes for 200K concepts from clinical findings.
+ `sbiobertresolve_loinc`: Map clinical NER entities to LOINC codes using `sbiobert`.
+ `sbluebertresolve_loinc`: Map clinical NER entities to LOINC codes using `sbluebert`.
+ `sbiobertresolve_HPO`: This model returns Human Phenotype Ontology (HPO) codes for phenotypic abnormalities encountered in human diseases. It also returns associated codes from the following vocabularies for each HPO code:

		* MeSH (Medical Subject Headings)
		* SNOMED
		* UMLS (Unified Medical Language System )
		* ORPHA (international reference resource for information on rare diseases and orphan drugs)
		* OMIM (Online Mendelian Inheritance in Man)

  *Related Notebook*: [Resolver Models](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb)

#### New feature on Deidentification Module
+ isRandomDateDisplacement(True): Be able to apply a random displacement on obfuscation dates. The randomness is based on the seed.
+ Fix random dates when the format is not correct. Now you can repeat an execution using a seed for dates. Random dates will be based on the seed.

#### Five new healthcare code mapping pipelines:

+ `icd10cm_umls_mapping`: This pretrained pipeline maps ICD10CM codes to UMLS codes without using any text data. You’ll just feed white space-delimited ICD10CM codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'icd10cm': ['M89.50', 'R82.2', 'R09.01'],
          'umls': ['C4721411', 'C0159076', 'C0004044']}

+ `mesh_umls_mapping`: This pretrained pipeline maps MeSH codes to UMLS codes without using any text data. You’ll just feed white space-delimited MeSH codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'mesh': ['C028491', 'D019326', 'C579867'],
         'umls': ['C0970275', 'C0886627', 'C3696376']}

+ `rxnorm_umls_mapping`: This pretrained pipeline maps RxNorm codes to UMLS codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'rxnorm': ['1161611', '315677', '343663'],
         'umls': ['C3215948', 'C0984912', 'C1146501']}

+ `rxnorm_mesh_mapping`: This pretrained pipeline maps RxNorm codes to MeSH codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding MeSH codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'rxnorm': ['1191', '6809', '47613'],
         'mesh': ['D001241', 'D008687', 'D019355']}

+ `snomed_umls_mapping`: This pretrained pipeline maps SNOMED codes to UMLS codes without using any text data. You’ll just feed white space-delimited SNOMED codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

      {'snomed': ['733187009', '449433008', '51264003'],
         'umls': ['C4546029', 'C3164619', 'C0271267']}

  *Related Notebook*: [Healthcare Code Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb)

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}