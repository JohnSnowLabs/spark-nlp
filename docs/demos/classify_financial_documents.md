---
layout: demopagenew
title: Classify Financial Documents - Finance NLP Demos & Notebooks
seotitle: 'Finance NLP: Classify Financial Documents - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /classify_financial_documents
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
        - subtitle: Classify Financial Documents - Live Demos & Notebooks
          activemenu: classify_financial_documents
      source: yes
      source: 
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
          - text: Colab
            type: blue_btn
            url: 
        - title: Financial News Classification 
          id: financial_news_classification        
          image: 
              src: /assets/images/Financial_News_Classification_new.svg
          image2: 
              src: /assets/images/Financial_News_Classification_new_f.svg
          excerpt: This model classifies financial news using multilabel categories.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/CLASSIFICATION_MULTILABEL/
          - text: Colab
            type: blue_btn
            url:         
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
          - text: Colab
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
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/BertForSequenceClassification.ipynb
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
          - text: Colab
            type: blue_btn
            url:  
        - title: Analyze sentiment in financial news
          id: analyze_sentiment_financial_news 
          image: 
              src: /assets/images/Analyze_sentiment_in_financial_news.svg
          image2: 
              src: /assets/images/Analyze_sentiment_in_financial_news_f.svg
          excerpt: This demo shows how sentiment can be identified (neutral, positive or negative) in financial news.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_EN_FINANCE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_FINANCE.ipynb
---