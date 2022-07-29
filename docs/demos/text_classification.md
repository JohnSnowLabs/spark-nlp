---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /text_classification
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Finance and Legal
      excerpt: Text Classification
      secheader: yes
      secheader:
        - title: Spark NLP for Finance and Legal
          subtitle: Text Classification
          activemenu: text_classification
      source: yes
      source: 
        - title: Classify Banking-related texts
          id: classify_banking_related_texts   
          image: 
              src: /assets/images/Classify_Banking-related_texts.svg
          image2: 
              src: /assets/images/Classify_Banking-related_texts_f.svg
          excerpt: This demo shows how to classify banking-related texts into 77 categories.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_BANKING/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/BertForSequenceClassification.ipynb
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
        - title: Financial News Classification  
          id: financial_news_classification     
          image: 
              src: /assets/images/Financial_News_Classification.svg
          image2: 
              src: /assets/images/Financial_News_Classification_f.svg
          excerpt: This model classifies financial news using multilabel categories.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/CLASSIFICATION_MULTILABEL/
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Classification of Bank Complaint Texts  
          id: classification_bank_complaint_texts      
          image: 
              src: /assets/images/Classification_of_Bank_Complaint_Text.svg
          image2: 
              src: /assets/images/Classification_of_Bank_Complaint_Text_f.svg
          excerpt: This model classifies the topic/class of a complaint about a bank-related product.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/COMPLAINT_CLASSIFICATION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/BertForSequenceClassification.ipynb  
---