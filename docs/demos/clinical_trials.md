---
layout: demopagenew
title: Clinical Trials - Healthcare NLP Demos & Notebooks
seotitle: 'Healthcare NLP: Clinical Trials - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /clinical_trials
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
        - subtitle: Clinical Trials - Live Demos & Notebooks
          activemenu: clinical_trials
      source: yes
      source: 
        - title: Extract Entities in Clinical Trial Abstracts  
          id: extract_entities_clinical_trial_abstracts     
          image: 
              src: /assets/images/ExtractEntitiesClinicalTrialAbstracts.svg
          excerpt: This model extracts to trial design, diseases, drugs, population, statistics, publication etc. relevant entities from clinical trial abstracts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CLINICAL_TRIALS_ABSTRACT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CLINICAL_TRIALS_ABSTRACT.ipynb
        - title: Recognize Concepts in Drug Development Trials
          id: recognize_concepts_in_drug_development_trials
          image: 
              src: /assets/images/Recognize_concepts_in_drug_development_trials.svg
          excerpt: This demo shows how to extract concepts related to drug development including Trial Groups, End Points and Hazard Ratio.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DRUGS_DEVELOPMENT_TRIALS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DRUGS_DEVELOPMENT_TRIALS.ipynb
        - title: Classify Randomized Clinical Trial (RCT) 
          id: classify_randomized_clinical_trial_rct   
          image: 
              src: /assets/images/Classify_Randomized_Clinical_Trial_RCT.svg
          excerpt: This demo shows a classifier that can classify whether an article is a randomized clinical trial (RCT) or not, as well as a classifier that can divide it into sections in abstracts of scientific articles on randomized clinical trials (RCT).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_RCT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLASSIFICATION_RCT.ipynb
        - title: Detect Covid-related clinical terminology 
          id: detect_covid_related_clinical_terminology
          image: 
              src: /assets/images/Detect_Covid-related_clinical_terminology.svg
          excerpt: This demo shows how Covid-related clinical terminology can be detected using a Spark NLP Healthcare NER model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_COVID/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_COVID.ipynb
---