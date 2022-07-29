---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /entity_recognition
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Finance and Legal 
      excerpt: Entity Recognition
      secheader: yes
      secheader:
        - title: Spark NLP for Finance and Legal
          subtitle: Entity Recognition
          activemenu: entity_recognition
      source: yes
      source: 
        - title: Recognize Ticker Alias in Financial texts
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
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TRAFFIC_DE.ipynb         
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
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_PROFESSIONS_ES.ipynb
        - title: Named Entity Recognition for (Brazilian) Portuguese Legal Texts 
          id: named_entity_recognition_brazilian_portuguese_legal_texts  
          image: 
              src: /assets/images/Named_Entity_Recognition_Brazilian_Portuguese_Legal_Texts.svg
          image2: 
              src: /assets/images/Named_Entity_Recognition_Brazilian_Portuguese_Legal_Texts_f.svg
          excerpt: Automatically identify entities such as Organization, Jurisprudence, Legislation, Person, Location, and Time, etc. in (Brazilian) Portuguese legal text. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_LEGAL_PT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_PT.ipynb
        - title: Name Entity Recognition on Financial Texts 
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
        - title: Extract Headers and Subheaders from Legal Documents 
          id: extract_headers_subheaders_from_legal_documents   
          image: 
              src: /assets/images/Extract_Headers_and_Subheaders_from_Legal_Documents.svg
          image2: 
              src: /assets/images/Extract_Headers_and_Subheaders_from_Legal_Documents_f.svg
          excerpt: This model uses Name Entity Recognition to detect HEADERS and SUBHEADERS with aims to detect the different sections of a legal document.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/LEGALNER_HEADERS/
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Extract Document Type, Parties, Aliases and Dates from Contracts 
          id: extract_document_type_parties_aliases_dates_contracts    
          image: 
              src: /assets/images/Extract_Document_Type.svg
          image2: 
              src: /assets/images/Extract_Document_Type_f.svg
          excerpt: This model uses Name Entity Recognition to extract DOC (Document Type), PARTY (An Entity signing a contract), ALIAS (the way a company is named later on in the document) and EFFDATE (Effective Date of the contract).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/LEGALNER_PARTIES/
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Extract Signers, Roles and Companies  
          id: extract_signers_roles_companies     
          image: 
              src: /assets/images/Extract_Signers_Roles.svg
          image2: 
              src: /assets/images/Extract_Signers_Roles_f.svg
          excerpt: This model uses Name Entity Recognition to extract SIGNING_PERSON (People signing a document), SIGNING_TITLE (the roles of those people in the company) and PARTY (Organizations).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/LEGALNER_SIGNERS/
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Extract Entities from Whereas clauses 
          id: extract_entities_whereas_clauses      
          image: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses.svg
          image2: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses_f.svg
          excerpt: This model uses Name Entity Recognition detect "Whereas" clauses and extract, from them, the SUBJECT, the ACTION and the OBJECT.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/LEGALNER_WHEREAS/
          - text: Colab Netbook
            type: blue_btn
            url:
        - title: Extract economic and social entities from Russian texts
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
---
