---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /analyze_medical_text_spanish
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Analyze Medical Texts in Spanish
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Analyze Medical Texts in Spanish
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
          - text: Colab Netbook
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
          - text: Colab Netbook
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
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR_ES.ipynb        
---