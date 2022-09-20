---
layout: demopagenew
title: Social Determinants - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Social Determinants - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /social_determinants
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
        - subtitle: Social Determinants - Live Demos & Notebooks
          activemenu: social_determinants
      source: yes
      source: 
        - title: Detect demographic information
          id: detect_demographic_information
          image: 
              src: /assets/images/Detect_demographic_information.svg
          image2: 
              src: /assets/images/Detect_demographic_information_f.svg
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
          image2: 
              src: /assets/images/Detect_demographics_and_vital_signs_using_rules_f.svg
          excerpt: Automatically detect demographic information as well as vital signs using our out-of-the-box Spark NLP Contextual Rules. Custom rules are very easy to define and run on your own data.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_PARSER
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_PARSER.ipynb
        - title: Identify gender using context and medical records
          id: identify_gender_using_context_and_medical_records
          image: 
              src: /assets/images/Detect_demographic_information.svg
          image2: 
              src: /assets/images/Detect_demographic_information_f.svg
          excerpt: Identify gender of a person by analyzing signs and symptoms using pretrained Spark NLP Classification model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_GENDER/
          - text: Colab
            type: blue_btn 
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLASSIFICATION_GENDER.ipynb
        - title: Detect risk factors
          id: detect_risk_factors
          image: 
              src: /assets/images/Detect_risk_factors.svg
          image2: 
              src: /assets/images/Detect_risk_factors_f.svg
          excerpt: Automatically identify risk factors such as <b>Coronary artery disease, Diabetes, Family history, Hyperlipidemia, Hypertension, Medications, Obesity, PHI, Smoking habits</b> in clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_RISK_FACTORS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_RISK_FACTORS.ipynb
        - title: Classify medical text according to PICO framework
          id: classify_medical_text_according
          image: 
              src: /assets/images/Classify_medical_text_according_to_PICO_framework.svg
          image2: 
              src: /assets/images/Classify_medical_text_according_to_PICO_framework_f.svg
          excerpt: 'Automatically classify medical text against PICO classes: CONCLUSIONS, DESIGN_SETTING, INTERVENTION, PARTICIPANTS, FINDINGS, MEASUREMENTS and AIMS.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_PICO/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb        
---
