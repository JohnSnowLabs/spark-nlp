---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /analyze_financial_models
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Analyze Financial Information
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Analyze Financial Information
          activemenu: analyze_financial_models
      source: yes
      source: 
        - title: Detect legal entities in German
          id: detect_legal_entities_german
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Automatically identify entities such as persons, judges, lawyers, countries, cities, landscapes, organizations, courts, trademark laws, contracts, etc. in German legal text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_DE.ipynb
        - title: Detect traffic information in German
          id: detect_traffic_information_in_text
          image: 
              src: /assets/images/Detect_traffic_information_in_text.svg
          image2: 
              src: /assets/images/Detect_traffic_information_in_text_f.svg
          excerpt: Automatically extract geographical location, postal codes, and traffic routes in German text using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_TRAFFIC_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb
        - title: Named Entity Recognition for (Brazilian) Portuguese Legal Texts
          id: named_entity_recognition_for_portuguese_legal_texts
          image: 
              src: /assets/images/Named_Entity_Recognition_for_Portuguese_Legal_Texts.svg
          image2: 
              src: /assets/images/Named_Entity_Recognition_for_Portuguese_Legal_Texts_f.svg
          excerpt: Recognize Organization, Jurisprudence, Legislation, Person, Location, and Time in legal texts (Brazilian Portuguese)
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_LENER/
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Detect professions and occupations in Spanish texts
          id: detect_professions_occupations_Spanish_texts 
          image: 
              src: /assets/images/Classify-documents.svg
          image2: 
              src: /assets/images/Classify-documents-w.svg
          excerpt: Automatically identify professions and occupations entities in Spanish texts using our pretrained Spark NLP for Healthcare model. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_PROFESSIONS_ES/ 
          - text: Colab Netbook
            type: blue_btn
            url:        
---