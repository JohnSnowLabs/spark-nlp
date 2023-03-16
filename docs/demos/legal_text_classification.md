---
layout: demopagenew
title: Classify Legal Texts - Legal NLP Demos & Notebooks
seotitle: 'Legal NLP: Classify Legal Texts - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /legal_text_classification
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
        - subtitle: Classify Legal Texts - Live Demos & Notebooks
          activemenu: legal_text_classification
      source: yes
      source: 
        - title: Classify hundreds types of clauses (Binary - clause detected or not)
          id: legal_clauses_classification    
          image: 
              src: /assets/images/Legal_Clauses_Classification.svg
          excerpt: These models check for specific clauses in legal texts, returning them (for example, "investments", "loans", etc. ) or “other” if the clause was not found.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/CLASSIFY_LEGAL_CLAUSES/
          - text: Colab
            type: blue_btn
            url:  
        - title: Classify 15 types of clauses (Multilabel)  
          id: classify_texts_15_types_legal_clauses     
          image: 
              src: /assets/images/Classify_texts_into_15_types_of_legal_clauses.svg
          excerpt: Using Multilabel Document Classification, where several classes can be assigned to a text, this demo will analyse and provide the best class or classes given an input text. This demo can be used to detect relevant clauses in a legal text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGMULTICLF_LEDGAR/
          - text: Colab
            type: blue_btn
            url:  
        - title: Classify Judgements Clauses 
          id: classify_judgements_clauses      
          image: 
              src: /assets/images/Classify_Judgements_Clauses.svg
          excerpt: These models analyze and identify if a clause is a decision, talks about a legal basis, a legitimate purpose, etc. and if an argument has been started by the ECHR, Commission/Chamber, the State, Third Parties, etc.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEG_JUDGEMENTS_CLF/
          - text: Colab
            type: blue_btn
            url: 
        - title: Classify Document into their Legal Type  
          id: classify_document_legal_type       
          image: 
              src: /assets/images/Classify_Document_into_their_Legal_Type.svg
          excerpt: This demo shows how to classify long texts / documents into a subsample of 8 different types.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/CLASSIFY_LEGAL_DOCUMENTS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/legal/CLASSIFY_LEGAL_DOCUMENTS.ipynb
        - title: Classify Swiss Judgements Documents  
          id: classify_swiss_judgements_documents       
          image: 
              src: /assets/images/Classify_Swiss_Judgements_Documents.svg
          excerpt: This demo shows how to classify Swiss Judgements documents in English, German, French, Italian into Civil Law, Insurance Law, Public Law, Social Law, Penal Law or Other.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGCLF_SWISS_JUDGEMENTS/
          - text: Colab
            type: blue_btn
            url: 
---