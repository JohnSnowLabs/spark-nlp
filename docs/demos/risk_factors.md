---
layout: demopagenew
title: Risk and Factors - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Risk and Factors - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /risk_factors
key: demo
nav_key: demo
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
        - subtitle: Risk Factors - Live Demos & Notebooks
          activemenu: risk_factors
      source: yes
      source:           
        - title: Calculate Medicare HCC Risk Score
          id: calculate_medicare_risk_score 
          image: 
              src: /assets/images/Calculate_Medicare_Risk_Score.svg
          excerpt: This demos shows how to calculate medical risk adjustment scores automatically using ICD codes of diseases.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/HCC_RISK_SCORE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb
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
        - title: Detect Smoking Status
          id: detect_smoking_status_entities
          image: 
              src: /assets/images/Detect_Smoking_Status.svg
          excerpt: This model detects the NER and assertion status of the related entities.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/SMOKING_STATUS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/SMOKING_STATUS.ipynb             
---