---
layout: demopage
title: Medical Risk Scoring
full_width: true
permalink: /medical_risk_scoring
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare
      excerpt: Medical Risk Scoring
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Medical Risk Scoring
          activemenu: medical_risk_scoring
      source: yes
      source: 
        - title: Calculate Medicare HCC Risk Score
          id: calculate_medicare_risk_score 
          image: 
              src: /assets/images/Calculate_Medicare_Risk_Score.svg
          image2: 
              src: /assets/images/Calculate_Medicare_Risk_Score_f.svg
          excerpt: This demos shows how to calculate medical risk adjustment scores automatically using ICD codes of diseases.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/HCC_RISK_SCORE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb
        - title: Extract Chunk Key Phrases 
          hide: yes
          id: extract_chunk_key_phrases  
          image: 
              src: /assets/images/Extract_Chunk_Key_Phrases.svg
          image2: 
              src: /assets/images/Extract_Chunk_Key_Phrases_f.svg
          excerpt: This demo shows how Chunk Key Phrases in medical texts can be extracted automatically using Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CHUNK_KEYWORD_EXTRACTOR/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/9.Chunk_Key_Phrase_Extraction.ipynb
        
---
