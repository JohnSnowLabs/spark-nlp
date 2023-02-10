---
layout: demopagenew
title: Mental Health - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Mental Health - John Snow Labs'
full_width: true
permalink: /mental_health
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
        - subtitle: Mental Health  - Live Demos & Notebooks
          activemenu: mental_health
      source: yes
      source: 
        - title: Identify Depression for Patient Posts
          id: depression_classifier_tweets 
          image: 
              src: /assets/images/Depression_Classifier_for_Tweets.svg
          excerpt: This demo shows a classifier that can classify whether tweets contain depressive text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/MENTAL_HEALTH_DEPRESSION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/MENTAL_HEALTH.ipynb
        - title: Identify Intimate Partner Violence from Patient Posts
          id: classify_intimate_partner_violence_tweet          
          image: 
              src: /assets/images/Classify_Intimate_Partner_Violence_Tweet.svg
          excerpt: This model involves the detection the potential IPV victims on social media platforms (in English tweets).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_PARTNER_VIOLENCE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb
        - title: Identify Stress from Patient Posts
          id: classify_stress_tweet        
          image: 
              src: /assets/images/Classify_Stress_Tweet.svg
          excerpt: This model can identify stress in social media (Twitter) posts in the self-disclosure category. The model finds whether a person claims he/she is stressed or not. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_STRESS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb
        - title: Identify the Source of Stress from Patient Posts
          id: identify_source_stress_patient_posts         
          image: 
              src: /assets/images/Identify_Source_Stress_Patient.svg
          excerpt: This demo shows how to classify source of emotional stress in text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_SOURCE_OF_STRESS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb
---
