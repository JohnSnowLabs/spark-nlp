---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /analyze_medical_text_german
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Analyze Medical Texts in German
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Analyze Medical Texts in German
          activemenu: analyze_medical_text_german
      source: yes
      source: 
        - title: Detect symptoms, treatments and other NERs in German
          id: detect_symptoms
          image: 
              src: /assets/images/Detect_causality_between_symptoms.svg
          image2: 
              src: /assets/images/Detect_causality_between_symptoms_f.svg
          excerpt: Automatically identify entities such as symptoms, diagnoses, procedures, body parts or medication in German clinical text using the pretrained Spark NLP clinical model ner_healthcare.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HEALTHCARE_DE.ipynb        
---