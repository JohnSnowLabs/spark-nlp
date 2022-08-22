---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /financial_text_classification
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Finance
      excerpt: Financial Text Classification
      secheader: yes
      secheader:
        - title: Spark NLP for Finance
          subtitle: Financial Text Classification
          activemenu: financial_text_classification
      source: yes
      source: 
        - title: Identify topics about banking
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
        - title: Classify Customer Support tickets (banking)  
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
        - title: ESG News Classification  
          id: esg_news_classification       
          image: 
              src: /assets/images/ESG_News_Classification.svg
          image2: 
              src: /assets/images/ESG_News_Classification_f.svg
          excerpt: This demo showcases ESG news classification, with 3-classes and 27-classes ESG models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINCLF_ESG/
          - text: Colab Netbook
            type: blue_btn
            url: 
        - title: Forward Looking Statements Classification 
          id: forward_looking_statements_classification       
          image: 
              src: /assets/images/Forward_Looking_Statements_Classification.svg
          image2: 
              src: /assets/images/Forward_Looking_Statements_Classification_f.svg
          excerpt: This demo shows how you can detect Forward Looking Statements in Financial Texts, as 10K filings or annual reports.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINCLF_FLS/
          - text: Colab Netbook
            type: blue_btn
            url: https://nlp.johnsnowlabs.com/ 
---