---
layout: demopagenew
title: Analyze Medical Texts in Spanish - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Analyze Medical Texts in Spanish - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /analyze_medical_text_spanish
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
        - subtitle: Analyze Medical Texts in Spanish - Live Demos & Notebooks
          activemenu: analyze_medical_text_spanish
      source: yes
      source: 
        - title: Detect Diagnoses And Procedures In Spanish
          id: detect-diagnoses-and-procedures-in-spanish
          image: 
              src: /assets/images/Detect_drugs_and_prescriptions.svg
          image2: 
              src: /assets/images/Detect_drugs_and_prescriptions_f.svg
          excerpt: Automatically identify diagnoses and procedures in Spanish clinical documents.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DIAG_PROC_ES/
          - text: Colab
            type: blue_btn
            url: https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DIAG_PROC_ES.ipynb
        - title: Resolve Clinical Health Information using the HPO taxonomy (Spanish) 
          id: hpo_coding_spanish
          image: 
              src: /assets/images/HPO_coding_Spanish.svg
          image2: 
              src: /assets/images/HPO_coding_Spanish_f.svg
          excerpt: Entity Resolver for Human Phenotype Ontology in Spanish
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_HPO_ES/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb
        - title: Detect Tumor Characteristics in Spanish medical texts
          id: detect_tumor_characteristics_spanish_medical_texts  
          image: 
              src: /assets/images/Detect_Tumor_Characteristics_in_Spanish_medical_texts.svg
          image2: 
              src: /assets/images/Detect_Tumor_Characteristics_in_Spanish_medical_texts_f.svg
          excerpt: This demo shows how to detect tumor characteristics (morphology) in Spanish medical texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_TUMOR_ES/  
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR_ES.ipynb
        - title: Map clinical terminology to SNOMED taxonomy in Spanish
          id: map_clinical_terminology_SNOMED_taxonomy_Spanish   
          image: 
              src: /assets/images/Map_clinical_terminology_to_SNOMED_taxonomy_in_Spanish.svg
          image2: 
              src: /assets/images/Map_clinical_terminology_to_SNOMED_taxonomy_in_Spanish_f.svg
          excerpt: This model maps healthcare information in Spanish to SNOMED codes using Entity Resolvers.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_SNOMED_ES
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_SNOMED_ES.ipynb
        - title: Deidentify Spanish texts
          id: deidentify_spanish_texts   
          image: 
              src: /assets/images/Detect_Tumor_Characteristics_in_Spanish_medical_texts.svg
          image2: 
              src: /assets/images/Detect_Tumor_Characteristics_in_Spanish_medical_texts_f.svg
          excerpt: This demo shows how to deidentify protected health information in Spanish medical texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_ES/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.2.Clinical_Deidentification_in_Spanish.ipynb
        - title: Detect PHI for Deidentification in Spanish
          id: detect_phi_deidentification_spanish    
          image: 
              src: /assets/images/Detect_risk_factors.svg
          image2: 
              src: /assets/images/Detect_risk_factors_f.svg
          excerpt: This demo shows how to detect Protected Health Information (PHI) that may need to be deidentified in Spanish.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DEID_ES/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.2.Clinical_Deidentification_in_Spanish.ipynb
        - title: Detection of disease mentions in Spanish tweets 
          id: detection_disease_mentions_spanish_tweets       
          image: 
              src: /assets/images/Detection_of_disease_mentions_in_Spanish_tweets.svg
          excerpt: This model extracts disease entities in Spanish tweets.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_NER_DISEASE_ES/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4TC.ipynb
---