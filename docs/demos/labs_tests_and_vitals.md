---
layout: demopagenew
title: Labs, Tests, and Vitals - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Labs, Tests, and Vitals - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /labs_tests_and_vitals
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
        - subtitle: Labs, Tests, and Vitals - Live Demos & Notebooks
          activemenu: labs_tests_and_vitals
      source: yes
      source:           
        - title: Classify Patient Demographics
          id: detect_demographic_information
          image: 
              src: /assets/images/Detect_demographic_information.svg
          excerpt: Automatically identify demographic information such as <b>Date, Doctor, Hospital, ID number, Medical record, Patient, Age, Profession, Organization, State, City, Country, Street, Username, Zip code, Phone number</b> in clinical documents using three of our pretrained Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb
        - title: Detect demographics and vital signs using rules
          id: detect_demographics_and_vital_signs_using_rules
          image: 
              src: /assets/images/Detect_demographics_and_vital_signs_using_rules.svg
          excerpt: Automatically detect demographic information as well as vital signs using our out-of-the-box Spark NLP Contextual Rules. Custom rules are very easy to define and run on your own data.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_PARSER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_PARSER.ipynb
        - title: Detect lab results
          id: detect_lab_results
          image: 
              src: /assets/images/Detect_lab_results.svg
          excerpt: Automatically identify <b>Lab test names</b> and <b>Lab results</b> from clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_LAB/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LAB.ipynb
        - title: Detect risk factors
          id: detect_risk_factors
          image: 
              src: /assets/images/Detect_risk_factors.svg
          excerpt: Automatically identify risk factors such as <b>Coronary artery disease, Diabetes, Family history, Hyperlipidemia, Hypertension, Medications, Obesity, PHI, Smoking habits</b> in clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_RISK_FACTORS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_RISK_FACTORS.ipynb
        - title: Detect clinical events
          id: detect_clinical_events
          image: 
              src: /assets/images/Detect_clinical_events.svg
          excerpt: Automatically identify a variety of clinical events such as <b>Problems, Tests, Treatments, Admissions</b> or <b>Discharges</b>, in clinical documents using two of our pretrained Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_EVENTS_CLINICAL.ipynb
        - title: Identify gender using context and medical records
          id: identify_gender_using_context_and_medical_records
          image: 
              src: /assets/images/Detect_demographic_information.svg
          excerpt: Identify gender of a person by analyzing signs and symptoms using pretrained Spark NLP Classification model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_GENDER/
          - text: Colab
            type: blue_btn 
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLASSIFICATION_GENDER.ipynb
        - title: Extract neurologic deficits related to NIH Stroke Scale (NIHSS)
          id: extract_neurologic_deficits_relatedNIH_stroke_scale 
          image: 
              src: /assets/images/Extract_neurologic_deficits_related_NIH_Stroke_Scale.svg
          excerpt: This demo shows how neurologic deficits can be extracted in accordance with their NIH Stroke Scale using a Spark NLP Healthcare NER model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_NIHSS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_NIHSS.ipynb
        - title: Identify relations between scale items and measurements according to NIHSS
          id: identify_relations_between_scale_items_their_measurements_according
          image: 
              src: /assets/images/Identify_relations_between_scale_items_measurements_according.svg
          excerpt: This demo shows how relations between scale items and their measurements can be identified according to NIHSS guidelines using a Spark NLP Healthcare RE model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_NIHSS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_NIHSS.ipynb     
---