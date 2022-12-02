---
layout: demopagenew
title: Middle Eastern Languages - Spark NLP Demos & Notebooks
seotitle: 'Spark NLP: Middle Eastern Languages - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /middle_eastern_languages
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
        - subtitle: Middle Eastern Languages - Live Demos & Notebooks
          activemenu: middle_eastern_languages
      source: yes
      source: 
        - title: Recognize entities in Turkish text
          id: recognize_entities_in_turkish_text
          image: 
              src: /assets/images/Flag_of_Turkey.png
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and multi-lingual Bert word embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_TR/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_TR.ipynb
        - title: Recognize entities in Arabic text 
          id: recognize_entities_in_arabic_text
          image: 
              src: /assets/images/arab.jpg
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_AR/ 
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/NER_AR.ipynb
        - title: Recognize entities in Urdu text
          id: recognize_entities_in_urdu_text
          image: 
              src: /assets/images/Flag_of_Pakistan.png
          excerpt: Recognize Persons, Locations and other entities using an out of the box pretrained Deep Learning model and language specific embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_UR/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/NER_UR.ipynb     
        - title: Analyze sentiment in Urdu movie reviews
          id: analyze_sentiment_in_urdu_movie_reviews
          image: 
              src: /assets/images/Flag_of_Pakistan.png
          excerpt: Detect the general sentiment expressed in a movie review or tweet by using our pretrained Spark NLP sentiment analysis model for Urdu language.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_UR/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb
        - title: Recognize entities in Persian text 
          id: recognize_entities_in_persian_text
          image: 
              src: /assets/images/Flag_of_Iran.png
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_FA/ 
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title: Recognize entities in Hebrew text 
          id: recognize_entities_in_hebrew_text
          image: 
              src: /assets/images/Israel.jpg
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_HE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/NER_HE.ipynb
        - title: Turkish News Classifier
          id: turkish_news_classifier
          image: 
              src: /assets/images/Healthcare_TurkishNewsClassifier.svg
          excerpt: Classify Turkish news text using our pre-trained model
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_TR_NEWS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_NEWS.ipynb
        - title: Turkish Cyberbullying Detection
          id: turkish_cyberbullying_detection
          image: 
              src: /assets/images/Turkish_Cyberbullying_Detection.svg
          excerpt: This demo shows how cyberbullying content can be automatically detected in Turkish text using Classifier DL model. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_TR_CYBERBULLYING/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_CYBERBULLYING.ipynb 
        - title: Analyze sentiment in Turkish texts
          id: analyze_sentiment_turkish_texts 
          image: 
              src: /assets/images/Analyze_sentiment_in_Turkish_texts.svg
          excerpt: This demo shows how sentiment can be identified (positive or negative) in Turkish texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_TR/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_TR_SENTIMENT.ipynb
        - title: Urdu news classifier 
          id: urdu_news_classifier  
          image: 
              src: /assets/images/Urdu_news_classifier.svg
          excerpt: This demo shows how to classify Urdu news into different categories, such as Science, Entertainment, etc.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_UR_NEWS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_UR_NEWS.ipynb
        - title: Urdu fake news classifier
          id: urdu_fake_news_classifier
          image: 
              src: /assets/images/Urdu_fake_news.svg
          excerpt: This demo shows how to detect fake Urdu news.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_UR_NEWS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_UR_NEWS.ipynb
        - title: Lemmatizer for Middle Eastern Languages
          id: lemmatizer_middle_mastern_languages 
          image: 
              src: /assets/images/Lemmatizer_for_Middle_Eastern_Languages.svg
          excerpt: This demo shows how to lemmatize documents of Middle Eastern languages.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/LEMMATIZER_MIDDLE_EAST/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb
---
