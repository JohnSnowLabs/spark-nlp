---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /financial_entity_recognition
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Finance
      excerpt: Financial Entity Recognition
      secheader: yes
      secheader:
        - title: Spark NLP for Finance
          subtitle: Financial Entity Recognition
          activemenu: financial_entity_recognition
      source: yes
      source:
        - title: Name Entity Recognition on financial texts 
          id: name_entity_recognition_financial_texts  
          image: 
              src: /assets/images/Name_Entity_Recognition_on_Financial_Texts.svg
          image2: 
              src: /assets/images/Name_Entity_Recognition_on_Financial_Texts_f.svg
          excerpt: This demo shows how you can extract the standard four entities (ORG, PER, LOC, MISC) from financial documents.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/NER_SEC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb
        - title: Extract Trading Symbols / Tickers
          id: recognize_ticker_alias_in_financial_texts 
          image: 
              src: /assets/images/Recognize_ticker_alias_in_financial_texts.svg
          image2: 
              src: /assets/images/Recognize_ticker_alias_in_financial_texts_f.svg
          excerpt: This demo shows how to extract ticker alias from financial texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_TICKER/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title: Extract Organizations and Products   
          id: extract_organizations_products        
          image: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses.svg
          image2: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses_f.svg
          excerpt: This model uses Name Entity Recognition to extract ORG (Organization names) and PRODUCT (Product names).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINNER_ORGPROD
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Extract economic and social entities in Russian
          id: extract_economical_social_entities_objects_government_documents       
          image: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses.svg
          image2: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses_f.svg
          excerpt: This model extracts entities such as ECO (economics), SOC (social) for economic and social entities, institutions of events, and also quantifiers (QUA), metrics (MET), etc. from Government documents in Russian.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FIN_NER_RUSSIAN_GOV
          - text: Colab Netbook
            type: blue_btn
            url:
---
