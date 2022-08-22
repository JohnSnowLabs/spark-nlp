---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /recognize_entitie
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP - English
      excerpt: Recognize Entities 
      secheader: yes
      secheader:
        - title: Spark NLP - English
          subtitle: Recognize Entities 
          activemenu: recognize_entitie
      source: yes
      source: 
        - title: Recognize 66 Entities in Text (Few-NERD)
          id: detect_wide_ranging_entities_market
          image: 
              src: /assets/images/Detect_wide-ranging_entities_in_the_Market.svg
          image2: 
              src: /assets/images/Detect_wide-ranging_entities_in_the_Market_c.svg
          excerpt: Detect 66 general entities such as art, newspaper, director, war, airport etc., using pretrained Spark NLP NER model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_FEW_NERD/
          - text: Colab Netbook
            type: blue_btn
            url : https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_FewNERD.ipynb
        - title: Recognize 18 Entities in Text (OntoNotes)
          id: recognize_more_entities_in_text
          image: 
              src: /assets/images/Spell_Checking.svg
          image2: 
              src: /assets/images/Spell_Checking_f.svg
          excerpt: Recognize over 18 entities such as <b>Countries, People, Organizations, Products, Events,</b> etc. using an out of the box pretrained NerDLApproach trained on the OntoNotes corpus.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_EN_18/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb
        - title: Detect Key Phrases (Unsupervised)
          id: detect_key_phrases
          image: 
              src: /assets/images/Detect_Key_Phrases.svg
          image2: 
              src: /assets/images/Detect_Key_Phrases_f.svg
          excerpt: Automatically detect key phrases in your text documents using out-of-the-box Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/KEYPHRASE_EXTRACTION
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/KEYPHRASE_EXTRACTION.ipynb
        - title: Find Text in a Document (Rule-Based)
          id: find_a_text_in_document
          image: 
              src: /assets/images/Find_in_Text.svg
          image2: 
              src: /assets/images/Find_in_Text_f.svg
          excerpt: Finds a text in document either by keyword or by regex expression.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/TEXT_FINDER_EN/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TEXT_FINDER_EN.ipynb
        - title: Recognize entities in text
          id: recognize_entities_in_text
          image: 
              src: /assets/images/Split_Clean_Text.svg
          image2: 
              src: /assets/images/Split_Clean_Text_f.svg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using out of the box pretrained Deep Learning models based on GloVe (glove_100d) and BERT (ner_dl_bert) word embeddings.
          actions:
          - text: Live Demo
            type: normal            
            url: https://demo.johnsnowlabs.com/public/NER_EN/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb          
        - title: Detect and normalize dates
          id: detect_and_normalize_dates
          image: 
              src: /assets/images/Detect_and_normalize_dates.svg
          image2: 
              src: /assets/images/Detect_and_normalize_dates_f.svg
          excerpt: Automatically detect key phrases expressing dates and normalize them with respect to a reference date.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/DATE_MATCHER/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/DATE_MATCHER.ipynb        
        - title: Detect Entities in tweets 
          id: detect_entities_tweets  
          image: 
              src: /assets/images/Detect_Entities_in_tweets.svg
          image2: 
              src: /assets/images/Detect_Entities_in_tweets_f.svg
          excerpt: This demo shows how to extract Named Entities, as PER, ORG or LOC, from tweets.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_BTC/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_BTC.ipynb
        - title: Recognize Restaurant Terminology 
          id: recognize_restaurant_terminology  
          image: 
              src: /assets/images/Recognize_restaurant_terminology.svg
          image2: 
              src: /assets/images/Recognize_restaurant_terminology_f.svg
          excerpt: This demo shows how to extract restaurant-related terminology from texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_RESTAURANT/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_RESTAURANT.ipynb
        - title: Recognize Time-related Terminology  
          id: recognize_time-related_terminology 
          image: 
              src: /assets/images/Recognize_time-related_terminology.svg
          image2: 
              src: /assets/images/Recognize_time-related_terminology_f.svg
          excerpt: This demo shows how to extract time-related terminology from texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_TIMEX_SEMEVAL/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb   
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
---
