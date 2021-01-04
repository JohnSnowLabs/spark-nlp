---
layout: demo
title: Spark NLP in Action
full_width: true
permalink: /demo
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: The most widely used NLP library in the enterprise
      excerpt: Backed by <b>O'Reilly's</b> most recent "AI Adoption in the Enterprise" survey in February
      tabheader: yes
      tabheader: 
        - title: Open Source <strong>Free</strong>
          url: opensource
          default: opensource
        - title: Languages <strong>Free</strong>
          url: languages
          default: languages
        - title: Healthcare
          url: healthcare
          default: healthcare
        - title: Spark OCR
          url: sparkocr
          default: sparkocr
        - title: De-identification
          url: deidentification
          default: deidentification
      opensource: yes
      opensource: 
        - title: Recognize entities in text
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
        - title: Classify documents
          image: 
              src: /assets/images/Classify-documents.svg
          image2: 
              src: /assets/images/Classify-documents-w.svg
          excerpt: Classify open-domain, fact-based questions into one of the following broad semantic categories <b>Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb
        - title: Analyze sentiment in movie reviews and tweets
          image: 
              src: /assets/images/imdb.svg
          image2: 
              src: /assets/images/imdb-w.svg
          excerpt: Detect the general sentiment expressed in a movie review or tweet by using our pretrained Spark NLP DL classifier.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_EN/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb
        - title: Detect emotions in tweets
          image: 
              src: /assets/images/Detect-emotions.svg
          image2: 
              src: /assets/images/Detect-emotions-w.svg
          excerpt: Automatically identify <b>Joy, Surprise, Fear, Sadness</b> in Tweets using out pretrained Spark NLP DL classifier.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_EN_EMOTION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_EMOTION.ipynb
        - title: Detect cyberbullying in tweets
          image: 
              src: /assets/images/twitter-2.svg
          image2: 
              src: /assets/images/twitter-2-w.svg
          excerpt: Identify <b>Racism, Sexism or Neutral</b> tweets using our pretrained emotions detector.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_EN_CYBERBULLYING/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_CYBERBULLYING.ipynb
        - title: Detect sarcastic tweets
          image: 
              src: /assets/images/Detect-sarcastic-tweets.svg
          image2: 
              src: /assets/images/Detect-sarcastic-tweets-w.svg
          excerpt: Checkout our sarcasm detection pretrained Spark NLP model. It is able to tell apart normal content from sarcastic content.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_EN_SARCASM/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_SARCASM.ipynb
        - title: Identify Fake news
          image: 
              src: /assets/images/fake-news.svg
          image2: 
              src: /assets/images/fake-news-w.svg
          excerpt: Determine if news articles are <b>Real</b> of <b>Fake</b>.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_FAKENEWS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_FAKENEWS.ipynb
        - title: Detect Spam messages
          image: 
              src: /assets/images/exclamation.svg
          image2: 
              src: /assets/images/exclamation-w.svg
          excerpt: Automatically identify messages as being regular messages or <b>Spam</b>.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_SPAM/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_SPAM.ipynb
        - title: Find a text in document
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
        - title: Grammar analysis & Dependency Parsing
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Visualize the syntactic structure of a sentence as a directed labeled graph where nodes are labeled with the part of speech tags and arrows contain the dependency tags.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/GRAMMAR_EN/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb
        - title: Split and clean text
          image: 
              src: /assets/images/Document_Classification.svg
          image2: 
              src: /assets/images/Document_Classification_f.svg
          excerpt: Spark NLP pretrained annotators allow an easy and straightforward processing of any type of text documents. This demo showcases our Sentence Detector, Tokenizer, Stemmer, Lemmatizer, Normalizer and Stop Words Removal.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/TEXT_PREPROCESSING/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TEXT_PREPROCESSING.ipynb
        - title: Spell check your text documents
          image: 
              src: /assets/images/spelling.svg
          image2: 
              src: /assets/images/spelling_f.svg
          excerpt: Spark NLP contextual spellchecker allows the quick identification of typos or spell issues within any text document.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SPELL_CHECKER_EN
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb
        - title: Detect Key Phrases
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
        - title: Detect similar sentences
          image: 
              src: /assets/images/Detect_similar_sentences.svg
          image2: 
              src: /assets/images/Detect_similar_sentences_f.svg
          excerpt: Automatically compute the similarity between two sentences using Spark NLP Universal Sentence Embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTENCE_SIMILARITY
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTENCE_SIMILARITY.ipynb
        - title: Detect toxic content in comments
          image: 
              src: /assets/images/Detect_Toxic_Comments.svg
          image2: 
              src: /assets/images/Detect_Toxic_Comments_f.svg
          excerpt: Automatically detect identity hate, insult, obscene, severe toxic, threat or toxic content in SM comments using our out-of-the-box Spark NLP Multiclassifier DL.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_MULTILABEL_TOXIC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_MULTILABEL_TOXIC.ipynb
        - title: Aspect based sentiment analysis for restaurants
          image: 
              src: /assets/images/Aspect_based_sentiment_analysis_for_restaurants.svg
          image2: 
              src: /assets/images/Aspect_based_sentiment_analysis_for_restaurants_f.svg
          excerpt: Automatically detect positive, negative and neutral aspects about restaurants from the written feedback given by reviewers.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/ASPECT_BASED_SENTIMENT_RESTAURANT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ABSA_Inference.ipynb
        - title: Detect sentences in text
          image: 
              src: /assets/images/Detect_sentences_in_text.svg
          image2: 
              src: /assets/images/Detect_sentences_in_text_f.svg
          excerpt: Detect sentences from general purpose text documents using a deep learning model capable of understanding noisy sentence structures.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTENCE_DETECTOR/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb
        - title: Detect and normalize dates
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
      languages: yes
      languages: 
        - title: Detect language
          image: 
              src: /assets/images/globe.svg
          image2: 
              src: /assets/images/globe_w.svg
          excerpt: Spark NLP Language Detector offers support for 20 different languages <b>Bulgarian, Czech, German, Greek, English, Spanish, Finnish, French, Croatian, Hungarian, Italy, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Swedish, Turkish, and Ukrainian</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/LANGUAGE_DETECTOR/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Language_Detector.ipynb
        - title: Recognize entities in English text
          image: 
              src: /assets/images/United_Kingdom.png
          image2: 
              src: /assets/images/United_Kingdom.png
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using out of the box pretrained Deep Learning models based on GloVe (glove_100d) and BERT (ner_dl_bert) word embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_EN_18/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb
        - title: Recognize entities in French text
          image: 
              src: /assets/images/French_flag.svg
          image2: 
              src: /assets/images/French_flag.svg
          excerpt: Recognize entities in French text
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_FR/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_FR.ipynb
        - title: Recognize entities in German text
          image: 
              src: /assets/images/German_flag.svg
          image2: 
              src: /assets/images/German_flag.svg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using an out of the box pretrained Deep Learning model and GloVe word embeddings (glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_DE.ipynb
        - title: Recognize entities in Italian text
          image: 
              src: /assets/images/Italian_flag.svg
          image2: 
              src: /assets/images/Italian_flag.svg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using an out of the box pretrained Deep Learning model and GloVe word embeddings (glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_IT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_IT.ipynb
        - title: Recognize entities in Norwegian text
          image: 
              src: /assets/images/norway-flag.jpg
          image2: 
              src: /assets/images/norway-flag.jpg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using 3 different out of the box pretrained Deep Learning models based on different GloVe word embeddings (glove_100d &amp; glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_NO/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_NO.ipynb
        - title: Recognize entities in Polish text
          image: 
              src: /assets/images/poland-flag.jpg
          image2: 
              src: /assets/images/poland-flag.jpg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using 3 different out of the box pretrained Deep Learning models based on different GloVe word embeddings (glove_100d &amp; glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_PL/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_PL.ipynb
        - title: Recognize entities in Portuguese text
          image: 
              src: /assets/images/flag-400.png
          image2: 
              src: /assets/images/flag-400.png
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using 3 different out of the box pretrained Deep Learning models based on different GloVe word embeddings (glove_100d &amp; glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_PT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_PT.ipynb
        - title: Recognize entities in Russian text
          image: 
              src: /assets/images/russia-flag.jpg
          image2: 
              src: /assets/images/russia-flag.jpg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using 3 different out of the box pretrained Deep Learning models based on different GloVe word embeddings (glove_100d &amp; glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_RU/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_RU.ipynb
        - title: Recognize entities in Spanish text
          image: 
              src: /assets/images/spanish-flag-small.png
          image2: 
              src: /assets/images/spanish-flag-small.png
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using 3 different out of the box pretrained Deep Learning models based on different GloVe word embeddings (glove_100d &amp; glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_ES/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_ES.ipynb
        - title: Recognize entities in Danish text
          image: 
              src: /assets/images/Flag_of_Denmark.png
          image2: 
              src: /assets/images/Flag_of_Denmark.png
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using an out of the box pretrained Deep Learning model and GloVe word embeddings (glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_DA/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title: Recognize entities in Swedish text
          image: 
              src: /assets/images/Flag_of_Sweden.jpg
          image2: 
              src: /assets/images/Flag_of_Sweden.jpg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using an out of the box pretrained Deep Learning model and GloVe word embeddings (glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_SV/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title: Recognize entities in Finnish text
          image: 
              src: /assets/images/flag-of-finland.jpg
          image2: 
              src: /assets/images/flag-of-finland.jpg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using an out of the box pretrained Deep Learning model and GloVe word embeddings (glove_300d).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_FI/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title: Prebuilt pipeline for entity recognition in Danish
          image: 
              src: /assets/images/Prebuilt_pipeline_for_entity_recognition_in_Danish.svg
          image2: 
              src: /assets/images/Prebuilt_pipeline_for_entity_recognition_in_Danish_f.svg
          excerpt: This SparkNLP out-of-the-box pipeline returns tokens, lemmas, pos, embeddings and NERs in one line of code. It automatically recognizes <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities in Danish text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/PP_EXPLAIN_DOCUMENT_DA/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/PP_EXPLAIN_DOCUMENT.ipynb
        - title: Prebuilt pipeline for entity recognition in Swedish
          image: 
              src: /assets/images/Prebuilt_pipeline_for_entity_recognition_in_Swedish.svg
          image2: 
              src: /assets/images/Prebuilt_pipeline_for_entity_recognition_in_Swedish_f.svg
          excerpt: This SparkNLP out-of-the-box pipeline returns tokens, lemmas, pos, embeddings and NERs in one line of code. It automatically recognizes <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities in Swedish text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/PP_EXPLAIN_DOCUMENT_SV/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/PP_EXPLAIN_DOCUMENT.ipynb
        - title: Prebuilt pipeline for entity recognition in Finnish
          image: 
              src: /assets/images/Prebuilt_pipeline_for_entity_recognition_in_Finnish.svg
          image2: 
              src: /assets/images/Prebuilt_pipeline_for_entity_recognition_in_Finnish_f.svg
          excerpt: This SparkNLP out-of-the-box pipeline returns tokens, lemmas, pos, embeddings and NERs in one line of code. It automatically recognizes <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities in Finnish text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/PP_EXPLAIN_DOCUMENT_FI/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/PP_EXPLAIN_DOCUMENT.ipynb
        - title: Recognize entities in Turkish text
          image: 
              src: /assets/images/Flag_of_Turkey.png
          image2: 
              src: /assets/images/Flag_of_Turkey.png
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and multi-lingual Bert word embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_TR/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_TR.ipynb
        - title: Recognize entities in Arabic text 
          image: 
              src: /assets/images/arab.jpg
          image2: 
              src: /assets/images/arab.jpg
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_AR/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Public_NER.ipynb
        - title: Recognize entities in Persian text 
          image: 
              src: /assets/images/Flag_of_Iran.png
          image2: 
              src: /assets/images/Flag_of_Iran.png
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_FA/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Public_NER.ipynb
        - title: Recognize entities in Hebrew text 
          image: 
              src: /assets/images/Israel.jpg
          image2: 
              src: /assets/images/Israel.jpg
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_HE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Public_NER.ipynb
        - title: Recognize entities in Japanese text 
          image: 
              src: /assets/images/Flag_of_Japan.png
          image2: 
              src: /assets/images/Flag_of_Japan.png
          excerpt: Recognize Persons, Locations and Organization entities using an out of the box pretrained Deep Learning model and language specific embeddings. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/NER_JA/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Public_NER.ipynb
      healthcare: yes
      healthcare: 
        - title: Detect signs and symptoms
          image: 
              src: /assets/images/Detect_signs_and_symptoms.svg
          image2: 
              src: /assets/images/Detect_signs_and_symptoms_f.svg
          excerpt: Automatically identify <b>Signs</b> and <b>Symptoms</b> in clinical documents using two of our pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_SIGN_SYMP/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_SIGN_SYMP.ipynb
        - title: Detect diagnosis and procedures
          image: 
              src: /assets/images/Detect_diagnosis_and_procedures.svg
          image2: 
              src: /assets/images/Detect_diagnosis_and_procedures_f.svg
          excerpt: Automatically identify diagnoses and procedures in clinical documents using the pretrained Spark NLP clinical model <b>ner_clinical.</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DIAG_PROC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DIAG_PROC.ipynb
        - title: Detect drugs and prescriptions
          image: 
              src: /assets/images/Detect_drugs_and_prescriptions.svg
          image2: 
              src: /assets/images/Detect_drugs_and_prescriptions_f.svg
          excerpt: Automatically identify <b>Drug, Dosage, Duration, Form, Frequency, Route,</b> and <b>Strength</b> details in clinical documents using three of our pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_POSOLOGY.ipynb
        - title: Detect risk factors
          image: 
              src: /assets/images/Detect_risk_factors.svg
          image2: 
              src: /assets/images/Detect_risk_factors_f.svg
          excerpt: Automatically identify risk factors such as <b>Coronary artery disease, Diabetes, Family history, Hyperlipidemia, Hypertension, Medications, Obesity, PHI, Smoking habits</b> in clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_RISK_FACTORS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_RISK_FACTORS.ipynb
        - title: Detect anatomical references
          image: 
              src: /assets/images/Detect_anatomical_references.svg
          image2: 
              src: /assets/images/Detect_anatomical_references_f.svg
          excerpt: Automatically identify <b>Anatomical System, Cell, Cellular Component, Anatomical Structure, Immaterial Anatomical Entity, Multi-tissue Structure, Organ, Organism Subdivision, Organism Substance, Pathological Formation</b> in clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_ANATOMY.ipynb
        - title: Detect demographic information
          image: 
              src: /assets/images/Detect_demographic_information.svg
          image2: 
              src: /assets/images/Detect_demographic_information_f.svg
          excerpt: Automatically identify demographic information such as <b>Date, Doctor, Hospital, ID number, Medical record, Patient, Age, Profession, Organization, State, City, Country, Street, Username, Zip code, Phone number</b> in clinical documents using three of our pretrained Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb
        - title: Detect clinical events
          image: 
              src: /assets/images/Detect_clinical_events.svg
          image2: 
              src: /assets/images/Detect_clinical_events_f.svg
          excerpt: Automatically identify a variety of clinical events such as <b>Problems, Tests, Treatments, Admissions</b> or <b>Discharges</b>, in clinical documents using two of our pretrained Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_EVENTS_CLINICAL.ipynb
        - title: Detect lab results
          image: 
              src: /assets/images/Detect_lab_results.svg
          image2: 
              src: /assets/images/Detect_lab_results_f.svg
          excerpt: Automatically identify <b>Lab test names</b> and <b>Lab results</b> from clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_LAB/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LAB.ipynb
        - title: Detect tumor characteristics
          image: 
              src: /assets/images/Detect_tumor_characteristics.svg
          image2: 
              src: /assets/images/Detect_tumor_characteristics_f.svg
          excerpt: Automatically identify <b>tumor characteristics</b> such as <b>Anatomical systems, Cancer, Cells, Cellular components, Genes and gene products, Multi-tissue structures, Organs, Organisms, Organism subdivisions, Simple chemicals, Tissues</b> from clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_TUMOR
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR.ipynb
        - title: Spell checking for clinical documents
          image: 
              src: /assets/images/Detect_clinical_events.svg
          image2: 
              src: /assets/images/Detect_clinical_events_f.svg
          excerpt: Automatically identify from clinical documents using our pretrained Spark NLP model <b>ner_bionlp.</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_SPELL_CHECKER.ipynb
        - title: SNOMED coding
          image: 
              src: /assets/images/Detect_signs_and_symptoms.svg
          image2: 
              src: /assets/images/Detect_signs_and_symptoms_f.svg
          excerpt: Automatically resolve the SNOMED code corresponding to the diseases and conditions mentioned in your health record using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_SNOMED
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_SNOMED.ipynb
        - title: ICDO coding
          image: 
              src: /assets/images/Detect_diagnosis_and_procedures.svg
          image2: 
              src: /assets/images/Detect_diagnosis_and_procedures_f.svg
          excerpt: Automatically detect the tumor in your healthcare records and link it to the corresponding ICDO code using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICDO
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICDO.ipynb
        - title: ICD10-CM coding 
          image: 
              src: /assets/images/Detect_risk_factors.svg
          image2: 
              src: /assets/images/Detect_risk_factors_f.svg
          excerpt: Automatically detect the pre and post op diagnosis, signs and symptoms or other findings in your healthcare records and automatically link them to the corresponding ICD10-CM code using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb
        - title: RxNORM coding
          image: 
              src: /assets/images/Detect_drugs_and_prescriptions.svg
          image2: 
              src: /assets/images/Detect_drugs_and_prescriptions_f.svg
          excerpt: Automatically detect the drugs and treatments names mentioned in your prescription or healthcare records and link them to the corresponding RxNORM codes using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_RXNORM
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_RXNORM.ipynb          
        - title: Detect demographics and vital signs using rules
          image: 
              src: /assets/images/Detect_demographics_and_vital_signs_using_rules.svg
          image2: 
              src: /assets/images/Detect_demographics_and_vital_signs_using_rules_f.svg
          excerpt: Automatically detect demographic information as well as vital signs using our out-of-the-box Spark NLP Contextual Rules. Custom rules are very easy to define and run on your own data.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_PARSER
          - text: Colab Netbook
            type: blue_btn
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_PARSER
        - title: Detect chemical compounds and genes
          image: 
              src: /assets/images/Detect_chemical_compounds_and_genes.svg
          image2: 
              src: /assets/images/Detect_chemical_compounds_and_genes_f.svg
          excerpt: Automatically detect all chemical compounds and gene mentions using our pretrained chemprot model included in Spark NLP for Healthcare.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMPROT_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMPROT_CLINICAL.ipynb
        - title: Detect genes and human phenotypes
          image: 
              src: /assets/images/Detect_genes_and_human_phenotypes.svg
          image2: 
              src: /assets/images/Detect_genes_and_human_phenotypes_f.svg
          excerpt: Automatically detect mentions of genes and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL.ipynb
        - title: Detect normalized genes and human phenotypes
          image: 
              src: /assets/images/Detect_normalized_genes_and_human_phenotypes.svg
          image2: 
              src: /assets/images/Detect_normalized_genes_and_human_phenotypes_f.svg
          excerpt: Automatically detect normalized mentions of genes (go) and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL.ipynb
        - title: ICD10 coding for German
          image: 
              src: /assets/images/Detect_diagnosis_and_procedures.svg
          image2: 
              src: /assets/images/Detect_diagnosis_and_procedures_f.svg
          excerpt: Automatically detect the pre and post op diagnosis, signs and symptoms in your German healthcare records and automatically link them to the corresponding ICD10-CM code using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICD10_GM_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_GM_DE.ipynb
        - title: Detect symptoms, treatments and other NERs in German
          image: 
              src: /assets/images/Detect_causality_between_symptoms.svg
          image2: 
              src: /assets/images/Detect_causality_between_symptoms_f.svg
          excerpt: Automatically identify entities such as symptoms, diagnoses, procedures, body parts or medication in German clinical text using the pretrained Spark NLP clinical model ner_healthcare.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HEALTHCARE_DE.ipynb
        - title: Detect legal entities German
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Automatically identify entities such as persons, judges, lawyers, countries, cities, landscapes, organizations, courts, trademark laws, contracts, etc. in German legal text using the pretrained Spark NLP models ner_legal.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/healthcare/NER_LEGAL_DE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_LEGAL_DE.ipynb
        - title: Adverse drug events tagger
          image: 
              src: /assets/images/Adverse_drug_events_tagger.svg
          image2: 
              src: /assets/images/Adverse_drug_events_tagger_f.svg
          excerpt: Automatic pipeline that tags documents as containing or not containing adverse events description, then identifies those events.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/PP_ADE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb
        - title: Identify diagnosis and symptoms assertion status
          image: 
              src: /assets/images/Identify_diagnosis_and_symptoms_assertion_status.svg
          image2: 
              src: /assets/images/Identify_diagnosis_and_symptoms_assertion_status_f.svg
          excerpt: Automatically detect if a diagnosis or a symptom is present, absent, uncertain or associated to other persons (e.g. family members).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ASSERTION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb
        - title: Detect cell structure, DNA, RNA and protein
          image: 
              src: /assets/images/Detect_cell_structure_DNA_RNA_and_protein.svg
          image2: 
              src: /assets/images/Detect_cell_structure_DNA_RNA_and_protein_f.svg
          excerpt: Automatically detect cell type, cell line, DNA and RNA information using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CELLULAR/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb
        - title: Link entities to Wikipedia pages
          image: 
              src: /assets/images/Link_entities_to_Wikipedia_pages.svg
          image2: 
              src: /assets/images/Link_entities_to_Wikipedia_pages_f.svg
          excerpt: Automatically disambiguate people’s names based on their context and link them to corresponding Wikipedia pages using out of the box Spark NLP pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DISAMBIGUATION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb
        - title: Detect posology relations
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Automatically identify relations between drugs, dosage, duration, frequency and strength using our pretrained clinical Relation Extraction (RE) model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_POSOLOGY/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_POSOLOGY.ipynb
        - title: Detect temporal relations for clinical events
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: 'Automatically identify three types of relations between clinical events: After, Before and Overlap using our pretrained clinical Relation Extraction (RE) model.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_EVENTS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL_EVENTS.ipynb
        - title: Detect causality between symptoms and treatment
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: Automatically identify relations between symptoms and treatment using our pretrained clinical Relation Extraction (RE) model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL.ipynb
        - title: Detect sentences in healthcare documents
          image: 
              src: /assets/images/Detect_sentences_in_healthcare_documents.svg
          image2: 
              src: /assets/images/Detect_sentences_in_healthcare_documents_f.svg
          excerpt: Automatically detect sentences in noisy healthcare documents with our pretrained Sentence Splitter DL model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/SENTENCE_DETECTOR_HC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb
        - title: Classify medical text according to PICO framework
          image: 
              src: /assets/images/Classify_medical_text_according_to_PICO_framework.svg
          image2: 
              src: /assets/images/Classify_medical_text_according_to_PICO_framework_f.svg
          excerpt: 'Automatically classify medical text against PICO classes: CONCLUSIONS, DESIGN_SETTING, INTERVENTION, PARTICIPANTS, FINDINGS, MEASUREMENTS and AIMS.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_PICO/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb
        - title: Detect chemical compounds
          image: 
              src: /assets/images/Detect_chemical_compounds.svg
          image2: 
              src: /assets/images/Detect_chemical_compounds_f.svg
          excerpt: Automatically detect all types of chemical compounds using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMICALS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb
        - title: Detect bacteria, plants, animals or general species
          image: 
              src: /assets/images/Detect_bacteria_plants_animals_or_general_species.svg
          image2: 
              src: /assets/images/Detect_bacteria_plants_animals_or_general_species_f.svg
          excerpt: Automatically detect bacteria, plants, animals, and other species using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_BACTERIAL_SPECIES/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb        
        - title: Detect traffic information in text
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
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb
      sparkocr: yes
      sparkocr: 
        - title: PDF to Text
          image: 
              src: /assets/images/PDF_to_Text.svg
          image2: 
              src: /assets/images/PDF_to_Text_f.svg
          excerpt: Extract text from generated/selectable PDF documents and keep the original structure of the document by using our out-of-the-box Spark OCR library.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PDF_TO_TEXT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TO_TEXT.ipynb
        - title: DICOM to Text
          image: 
              src: /assets/images/DICOM_to_Text.svg
          image2: 
              src: /assets/images/DICOM_to_Text_f.svg
          excerpt: Recognize text from DICOM format documents. This feature explores both to the text on the image and to the text from the metadata file.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DICOM_TO_TEXT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DICOM_TO_TEXT.ipynb
        - title: Image to Text
          image: 
              src: /assets/images/Image_to_Text.svg
          image2: 
              src: /assets/images/Image_to_Text_f.svg
          excerpt: Recognize text in images and scanned PDF documents by using our out-of-the-box Spark OCR library.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/IMAGE_TO_TEXT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/IMAGE_TO_TEXT.ipynb
        - title: Remove background noise from scanned documents
          image: 
              src: /assets/images/remove_bg.svg
          image2: 
              src: /assets/images/remove_bg_f.svg
          excerpt: Removing the background noise in a scanned document will highly improve the results of the OCR. Spark OCR is the only library that allows you to finetune the image preprocessing for excellent OCR results.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/BG_NOISE_REMOVER/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/BG_NOISE_REMOVER.ipynb
        - title: Correct skewness in scanned documents
          image: 
              src: /assets/images/correct.svg
          image2: 
              src: /assets/images/correct_f.svg
          excerpt: Correct the skewness of your scanned documents will highly improve the results of the OCR. Spark OCR is the only library that allows you to finetune the image preprocessing for excellent OCR results.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/SKEW_CORRECTION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/SKEW_CORRECTION.ipynb
        - title: Recognize text in natural scenes
          image: 
              src: /assets/images/Frame.svg
          image2: 
              src: /assets/images/Frame_f.svg
          excerpt: By using image segmentation and preprocessing techniques Spark OCR recognizes and extracts text from natural scenes.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/NATURAL_SCENE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/NATURAL_SCENE.ipynb
        - title: Recognize entities in scanned PDFs
          image: 
              src: /assets/images/Recognize_text_in_natural_scenes.svg
          image2: 
              src: /assets/images/Recognize_text_in_natural_scenes_f.svg
          excerpt: 'End-to-end example of regular NER pipeline: import scanned images from cloud storage, preprocess them for improving their quality, recognize text using Spark OCR, correct the spelling mistakes for improving OCR results and finally run NER for extracting entities.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PDF_TEXT_NER/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TEXT_NER.ipynb
        - title: Extract tables from PDFs
          image: 
              src: /assets/images/Extract_tables_from_PDFs.svg
          image2: 
              src: /assets/images/Extract_tables_from_PDFs_f.svg
          excerpt: Extract tables from selectable PDF documents with the new features offered by Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PDF_TEXT_TABLE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TEXT_TABLE.ipynb
      deidentification: yes
      deidentification: 
        - title: Deidentify structured data
          image: 
              src: /assets/images/Deidentify_structured_data.svg
          image2: 
              src: /assets/images/Deidentify_structured_data_f.svg
          excerpt: Deidentify PHI information from structured datasets using out of the box Spark NLP functionality that enforces GDPR and HIPPA compliance, while maintaining linkage of clinical data across files.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DEID_EHR_DATA
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_EHR_DATA.ipynb
        - title: Deidentify free text documents
          image: 
              src: /assets/images/Deidentify_free_text_documents.svg
          image2: 
              src: /assets/images/Deidentify_free_text_documents_f.svg
          excerpt: Deidentify free text documents by either masking or obfuscating PHI information using out of the box Spark NLP models that enforce GDPR and HIPPA compliance.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT.ipynb
        - title: Deidentify DICOM documents
          image: 
              src: /assets/images/Deidentify_DICOM_documents.svg
          image2: 
              src: /assets/images/Deidentify_DICOM_documents_f.svg
          excerpt: Deidentify DICOM documents by masking PHI information on the image and by either masking or obfuscating PHI from the metadata.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_DICOM_IMAGE
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_DICOM_IMAGE.ipynb
        - title: De-identify PDF documents - HIPAA Compliance
          image: 
              src: /assets/images/Deidentify_PDF_documents.svg
          image2: 
              src: /assets/images/Deidentify_PDF_documents_f.svg
          excerpt: De-identify PDF documents using HIPAA guidelines by masking PHI information using out of the box Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_PDF_HIPAA
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_PDF.ipynb
        - title: De-identify PDF documents - GDPR Compliance
          image: 
              src: /assets/images/Deidentify_PDF_documents.svg
          image2: 
              src: /assets/images/Deidentify_PDF_documents_f.svg
          excerpt: De-identify PDF documents using GDPR guidelines by anonymizing PHI information using out of the box Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_PDF_GDPR
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_PDF.ipynb
---
