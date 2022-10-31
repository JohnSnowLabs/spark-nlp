---
layout: demopagenew
title: Drugs & Adverse Events - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Drugs & Adverse Events - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /drug_adverse_events
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
        - subtitle: Drugs & Adverse Events - Live Demos & Notebooks
          activemenu: drug_adverse_events
      source: yes
      source:           
        - title: Detect drugs and prescriptions
          id: detect_drugs_and_prescriptions
          image: 
              src: /assets/images/Detect_drugs_and_prescriptions.svg
          excerpt: Automatically identify <b>Drug, Dosage, Duration, Form, Frequency, Route,</b> and <b>Strength</b> details in clinical documents using three of our pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_POSOLOGY.ipynb
        - title: Detect posology relations
          id: detect_posology_relations
          image: 
              src: /assets/images/Grammar_Analysis.svg
          excerpt: Automatically identify relations between drugs, dosage, duration, frequency and strength using our pretrained clinical Relation Extraction (RE) model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_POSOLOGY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_POSOLOGY.ipynb
        - title: Extract Drugs and Chemicals
          id: extract_names_of_drugs_chemicals 
          image: 
              src: /assets/images/Extract_the_Names_of_Drugs_Chemicals.svg
          excerpt: This demo shows how Names of Drugs & Chemicals can be detected using a Spark NLP Healthcare NER model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMD/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMD.ipynb
        - title: Identify Relations Between Drugs and Adversary Events
          id: identify_relations_between_drugs_and_ade
          image: 
              src: /assets/images/Identify_relations_between_drugs_and_ade.svg
          excerpt: This demo shows how to detect relations between drugs and adverse reactions caused by them.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_ADE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_ADE.ipynb
        - title: Extract conditions and benefits from drug reviews
          id: extract_conditions_benefits_drug_reviews 
          image: 
              src: /assets/images/Extract_conditions_and_benefits_from_drug_reviews.svg
          excerpt: This model shows how to extract conditions and benefits from drug reviews.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_SUPPLEMENT_CLINICAL/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_SUPPLEMENT_CLINICAL.ipynb
        - title: Detect Drug Chemicals 
          id: detect_drug_chemicals   
          image: 
              src: /assets/images/Detect_Drug_Chemicals.svg
          excerpt: Automatically identify drug chemicals in clinical documents using the pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DRUGS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DRUGS.ipynb
        - title: Detect ADE-related texts
          id: detect_ade_related_texts   
          image: 
              src: /assets/images/Detect_ADE_related_texts.svg
          excerpt: This model classifies texts as containing or not containing adverse drug events description.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_ADE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb    
---