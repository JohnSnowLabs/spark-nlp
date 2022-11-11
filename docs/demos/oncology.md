---
layout: demopagenew
title: Oncology - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Oncology - John Snow Labs'
full_width: true
permalink: /oncology
key: demo
article_header:
  type: demo
license: false
mode: immersivebg
show_edit_on_github: false
show_date: false
data:
  sections:  
    - secheader: yes
      secheader:
        - subtitle: Oncology - Live Demos & Notebooks
          activemenu: oncology
      source: yes
      source: 
        - title: Explore Oncology Notes with Spark NLP Models
          id: explore_oncology_notes_spark_models
          image: 
              src: /assets/images/Detect_Oncological_Concepts.svg
          excerpt: This demo shows how oncological terms can be detected using Spark NLP Healthcare NER, Assertion Status, and Relation Extraction models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ONCOLOGY/
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb
        - title: Detect oncological & biological concepts
          id: detect_tumor_characteristics
          image: 
              src: /assets/images/Detect_tumor_characteristics.svg
          excerpt: Automatically identify <b>oncological</b> and <b>biological</b> entities such as <b>Amino_acids, Anatomical systems, Cancer, Cells or Cellular components</b> using our pertained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_TUMOR
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR.ipynb  
        - title: Resolve Oncology terminology using the ICD-O taxonomy
          id: resolve_oncology_terminology_using_icdo_taxonomy
          image: 
              src: /assets/images/Resolve_Oncology_terminology.svg
          excerpt: This model maps oncology terminology to ICD-O codes using Entity Resolvers.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICDO/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICDO.ipynb
        - title: Extract the biomarker information
          id: extract_biomarker_information 
          image: 
              src: /assets/images/Extract_brands_from_visual_documents.svg
          excerpt: This demo shows how biomarkers, therapies, oncological, and other general concepts can be extracted using Spark NLP Healthcare NER model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_BIOMARKER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BIOMARKER.ipynb        
        - title: Detect Relation Extraction between Granular Oncological entity types
          id: detect_relation_extraction_granular_oncological_entity_types   
          image: 
              src: /assets/images/Detect_Relation_Extraction_between_Granular_Oncological_entity_types.svg
          excerpt: This demo shows how to identify relations between Clinical entities, Tumor mentions, Anatomical entities, Tests, and their results using pretrained Oncology Relation Extraction (RE) models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY_GRANULAR/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Detect Relation Extraction between different Oncological entity types
          id: detect_relation_extraction_different_oncological_entity_types   
          image: 
              src: /assets/images/Detect_Relation_Extraction_between_different_Oncological_entity_types.svg
          excerpt: This demo shows how to identify relations between Biomarkers, Anatomical Entities, Tumor Size, Tumor Finding, Date, Clinical Entities, Tests, and their corresponding results using pretrained Oncology Relation Extraction (RE) models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Identify oncology entities related to treatments different from clinical texts
          id: identify_oncology_entities_related_treatments_different_clinical_texts   
          image: 
              src: /assets/images/Identify_oncology_entities_related_to_treatments_different_from_clinical_texts.svg
          excerpt: This demo shows how to extract entities related to Oncology Therapies, Mentions of Treatments, posology information, Tumor Size, and Cancer Therapies using four pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_TREATMENT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Identify Oncology Tests, Biomarkers and their Results
          id: identify_oncology_tests_biomarkers_their_results   
          image: 
              src: /assets/images/Identify_Oncology_Tests.svg
          excerpt: This demo shows how to extract entities’ Pathology Tests, Imaging Tests, mentions of Biomarkers, and their results from oncology texts using pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_TESTS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Identify Oncology Entities Related to Diagnosis from Clinical Texts
          id: identify_oncology_entities_related_diagnosis_clinical_texts   
          image: 
              src: /assets/images/Identify_Oncology_Entities_Related_to_Diagnosis_from_Clinical_Text.svg
          excerpt: This demo shows how to extract more than 40 Oncology-related entities including those related to Cancer diagnosis, Staging information, Tumors, Lymph Nodes, and Metastases.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_DIAGNOSIS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Identify Demographic Information from Oncology Texts
          id: identify_demographic_information_from_oncology_texts   
          image: 
              src: /assets/images/Identify_Demographic_Information_from_Oncology_Text.svg
          excerpt: This demo shows how to extract Demographic information, Age, Gender, and Smoking status from oncology texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_DEMOGRAPHICS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Identify Anatomical Entities from Clinical Texts
          id: identify_anatomical_entities_from_clinical_texts   
          image: 
              src: /assets/images/Іdentify_Anatomical_Entities_from_Clinical_Text.svg
          excerpt: This demo shows how to extract anatomical entities using unspecific labels or granular labels from clinical texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_ANATOMY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Detect Assertion Status from Demographic Entities
          id: detect_assertion_status_from_demographic_entities   
          image: 
              src: /assets/images/Detect_Assertion_Status_from_Demographic_Entities.svg
          excerpt: This demo shows how to detect if a demographic entity refers to the patient or to someone else.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ASSERTION_ONCOLOGY_DEMOGRAPHIC/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
        - title: Detect Assertion Status from Clinics Entities 
          id: detect_assertion_status_from_clinics_entities   
          image: 
              src: /assets/images/Detect_Assertion_Status_from_Clinics_Entities.svg
          excerpt: This demo shows how to detect the assertion status of entities related to oncology (including diagnoses, therapies, and tests).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ASSERTION_ONCOLOGY_CLINICS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/27.Oncology_Model.ipynb
---