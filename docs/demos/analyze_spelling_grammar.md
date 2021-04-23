---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /analyze_spelling_grammar
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP - English
      excerpt: Analyze Spelling & Grammar 
      secheader: yes
      secheader:
        - title: Spark NLP - English
          subtitle: Analyze Spelling & Grammar 
          activemenu: analyze_spelling_grammar
      source: yes
      source: 
        - title:  Evaluate sentence grammar
          id: evaluate_sentence_grammar
          image: 
              src: /assets/images/Find_in_Text.svg
          image2: 
              src: /assets/images/Find_in_Text_f.svg
          excerpt: Classify a sentence as grammatically correct or incorrect.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/SENTENCE_GRAMMAR/
          - text: Colab Netbook
            type: blue_btn
            url:  https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb
        - title: Grammar analysis & Dependency Parsing
          id: grammar_analysis_dependency_parsing
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
        - title: Spell check your text documents
          id: spell_check_your_text_documents
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
        - title: Detect sentences in text
          id: detect_sentences_in_text
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
        - title: Split and clean text
          id: split_and_clean_text
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
---
