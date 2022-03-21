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
        - title:  Correct Sentences Grammar
          id: correct_sentences_grammar 
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: This demo shows how to correct grammatical errors in texts.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/T5_GRAMMAR/
          - text: Colab Netbook
            type: blue_btn
            url:  https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb    
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
        - title: Linguistic transformations on texts
          hide: yes
          id: linguistic_transformations_texts
          image: 
              src: /assets/images/Text_generation_for_linguistics.svg
          image2: 
              src: /assets/images/Text_generation_for_linguistics_f.svg
          excerpt: This demo shows how to correct grammatical errors and how to implement formal-informal and active-passive sentence conversions.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/T5_LINGUISTIC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb               
        - title:  Switch Between Active and Passive voice
          id: switch_between_active_passive_voice  
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: This demo shows how to switch between active and passive sentences.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/T5_ACTIVE_PASSIVE/
          - text: Colab Netbook
            type: blue_btn
            url:  https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb
        - title:  Switch Between Informal and Formal style
          id: switch_between_informal_formal  
          image: 
              src: /assets/images/Grammar_Analysis.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: This demo shows how to switch between texts written in formal and informal style.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/T5_FORMAL_INFORMAL/
          - text: Colab Netbook
            type: blue_btn
            url:  https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_LINGUISTIC.ipynb
        - title:  Evaluate Sentence Grammar
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
---
