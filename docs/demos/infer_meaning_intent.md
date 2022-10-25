---
layout: demopagenew
title: Infer Meaning & Intent - Spark NLP Demos & Notebooks
seotitle: 'Spark NLP: Infer Meaning & Intent - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /infer_meaning_intent
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
        - subtitle: Infer Meaning & Intent - Live Demos & Notebooks 
          activemenu: infer_meaning_intent
      source: yes
      source:         
        - title:  Understand intent and actions in general commands
          id: understand_intent_and_actions_in_general_commands
          image: 
              src: /assets/images/Split_Clean_Text.svg
          excerpt: Extract intents in general commands related to music, restaurants, movies.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/NER_CLS_SNIPS
          - text: Colab
            type: blue_btn
            url:  https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title:  Infer word meaning from context
          id: infer_word_meaning_from_context
          image: 
              src: /assets/images/Grammar_Analysis.svg
          excerpt: Compare the meaning of words in two different sentences and evaluate ambiguous pronouns.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/CONTEXTUAL_WORD_MEANING/
          - text: Colab
            type: blue_btn
            url:  https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb
        - title:  Assess relationship between two sentences
          id: assess_relationship_between_two_sentences
          image: 
              src: /assets/images/Spell_Checking.svg
          excerpt: Evaluate the relationship between two sentences or text fragments to identify things such as contradictions, entailments and premises & hypotheses
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/SENTENCE_RELATIONS/
          - text: Colab
            type: blue_btn
            url:  https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb
        - title: Detect similar sentences
          id: detect_similar_sentences
          image: 
              src: /assets/images/Detect_similar_sentences.svg
          excerpt: Automatically compute the similarity between two sentences using Spark NLP Universal Sentence Embeddings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTENCE_SIMILARITY
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTENCE_SIMILARITY.ipynb
        - title:  Automatically answer questions
          hide: yes
          id: automatically_answer_questions
          image: 
              src: /assets/images/spelling.svg
          excerpt: Automatically generate answers to questions with & without context
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/QUESTION_ANSWERING/
          - text: Colab
            type: blue_btn
            url:  https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5TRANSFORMER.ipynb
        - title: Understand questions about Airline Traffic
          id: understand_questions_about_airline_traffic
          image: 
              src: /assets/images/Detect_diagnosis_and_procedures.svg
          excerpt: Automatically detect key entities related to airline traffic, such as departure and arrival times and locations.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/NER_CLS_ATIS
          - text: Colab
            type: blue_btn
            url:  https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb
        - title: Extract graph entities and relations 
          id: extract_graphs_from_text 
          image: 
              src: /assets/images/Extract_Graphs_in_a_Text.svg
          excerpt: This demo shows how knowledge graphs entities and relations can be extracted from texts.
          actions:
          - text: Live Demo
            type: normal
            url:  https://demo.johnsnowlabs.com/public/GRAPH_RE/
          - text: Colab
            type: blue_btn
            url:  https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Graph_RE.ipynb
        - title: SQL Query Generation 
          id: sql_query_generation  
          image: 
              src: /assets/images/Sql_query_generation.svg
          excerpt: This demo shows how to generate SQL code from natural language text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/T5_SQL/ 
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/T5_SQL.ipynb  
        - title: Coreference Resolution
          id: coreference_resolution  
          image: 
              src: /assets/images/Sql_query_generation.svg
          excerpt: This demo shows how to identify expressions that refer to the same entity in a text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/COREFERENCE_RESOLUTION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/public/COREFERENCE_RESOLUTION.ipynb
---