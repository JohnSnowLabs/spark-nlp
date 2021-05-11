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
        - title: Recognize more entities in text
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
        - title: Detect Key Phrases
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
        - title: Find a text in a document
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
---
