---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /legal_text_classification
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Legal
      excerpt: Legal Text Classification
      secheader: yes
      secheader:
        - title: Spark NLP for Legal
          subtitle: Legal Text Classification
          activemenu: legal_text_classification
      source: yes
      source: 
        - title: Legal Clauses Classification 
          id: legal_clauses_classification    
          image: 
              src: /assets/images/Legal_Clauses_Classification.svg
          image2: 
              src: /assets/images/Legal_Clauses_Classification_f.svg
          excerpt: These models check for specific clauses in legal texts, returning them (for example, "investments", "loans", etc. ) or “other” if the clause was not found.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/CLASSIFY_LEGAL_CLAUSES/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb