---
layout: demopagenew
title: Recognize Legal Entities - Legal NLP Demos & Notebooks
seotitle: 'Legal NLP: Recognize Legal Entities - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /legal_entity_recognition
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
        - subtitle: Recognize Legal Entities - Live Demos & Notebooks
          activemenu: legal_entity_recognition
      source: yes
      source:
        - title: Extract Document Type, Parties, Aliases and Dates 
          id: extract_document_type_parties_aliases_dates_contracts    
          image: 
              src: /assets/images/Extract_Document_Type.svg
          image2: 
              src: /assets/images/Extract_Document_Type_f.svg
          excerpt: This model uses Name Entity Recognition to extract DOC (Document Type), PARTY (An Entity signing a contract), ALIAS (the way a company is named later on in the document) and EFFDATE (Effective Date of the contract).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGALNER_PARTIES/
          - text: Colab
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
            url: https://demo.johnsnowlabs.com/legal/LEGALNER_SIGNERS/
          - text: Colab
            type: blue_btn
            url:
        - title: Extract headers and subheaders 
          id: extract_headers_subheaders_from_legal_documents   
          image: 
              src: /assets/images/Extract_Headers_and_Subheaders_from_Legal_Documents.svg
          image2: 
              src: /assets/images/Extract_Headers_and_Subheaders_from_Legal_Documents_f.svg
          excerpt: This model uses Name Entity Recognition to detect HEADERS and SUBHEADERS with aims to detect the different sections of a legal document.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGALNER_HEADERS/
          - text: Colab
            type: blue_btn
            url: 
        - title: Extract entities in Whereas clauses 
          id: extract_entities_whereas_clauses      
          image: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses.svg
          image2: 
              src: /assets/images/Extract_Entities_from_Whereas_clauses_f.svg
          excerpt: This model uses Name Entity Recognition detect "Whereas" clauses and extract, from them, the SUBJECT, the ACTION and the OBJECT.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGALNER_WHEREAS/
          - text: Colab
            type: blue_btn
            url:
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
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_DE.ipynb
        - title: Detect legal entities in Portuguese
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
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_PT.ipynb
        - title: Extract Parties obligations in a Legal Agreement 
          id: extract_parties_obligations_legal_agreement   
          image: 
              src: /assets/images/Extract_Parties_obligations_in_a_Legal_Agreement.svg
          image2: 
              src: /assets/images/Extract_Parties_obligations_in_a_Legal_Agreement_f.svg
          excerpt: Automatically identify entities such as Organization, Jurisprudence, Legislation, Person, Location, and Time, etc. in (Brazilian) Portuguese legal text. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGALNER_OBLIGATIONS/
          - text: Colab
            type: blue_btn
            url: 
        - title: Identify Companies and their aliases in legal texts
          id: identify_companies_their_aliases_legal_texts        
          image: 
              src: /assets/images/Identify_Companies_and_their_aliases_in_legal_texts.svg
          image2: 
              src: /assets/images/Identify_Companies_and_their_aliases_in_legal_texts_f.svg
          excerpt: This model uses Entity Recognition to identify ORG (Companies), their ALIAS (other names the company uses in the contract/agreement) and company PRODUCTS.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGALNER_ALIAS/
          - text: Colab
            type: blue_btn
            url:
---
