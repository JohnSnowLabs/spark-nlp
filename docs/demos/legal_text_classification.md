---
layout: demopagenew
title: Classify Legal Texts - Legal NLP Demos & Notebooks
seotitle: 'Legal NLP: Classify Legal Texts - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /legal_text_classification
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
        - subtitle: Classify Legal Texts - Live Demos & Notebooks
          activemenu: legal_text_classification
      source: yes
      source: 
        - title: Classify 275 types of clauses (Binary - clause detected or not)
          id: legal_clauses_classification    
          image: 
              src: /assets/images/Legal_Clauses_Classification.svg
          image2: 
              src: /assets/images/Legal_Clauses_Classification_f.svg
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
          image2: 
              src: /assets/images/Classify_texts_into_15_types_of_legal_clauses_f.svg
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
---