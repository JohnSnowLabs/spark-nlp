---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /classify_medical_texts
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Classify Medical Texts
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Classify Medical Texts
          activemenu: classify_medical_texts
      source: yes
      source: 
        - title: PICO Classifier
          id: pico_classifier 
          image: 
              src: /assets/images/Classify-documents.svg
          image2: 
              src: /assets/images/Classify-documents-w.svg
          excerpt: This demo shows how to classify medical texts in accordance with PICO Components.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_PICO/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb
        - title: Gender Classifier
          id: gender_classifier  
          image: 
              src: /assets/images/Classify-documents.svg
          image2: 
              src: /assets/images/Classify-documents-w.svg
          excerpt: This demo shows how to identify gender from medical texts in accordance with the gender of the patient.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_GENDER/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLASSIFICATION_GENDER.ipynb
        - title: Detect ADE-related texts
          id: detect_ade_related_texts   
          image: 
              src: /assets/images/Detect_ADE_related_texts.svg
          image2: 
              src: /assets/images/Detect_ADE_related_texts_f.svg
          excerpt: This model classifies texts as containing or not containing adverse drug events description.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_ADE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb
---