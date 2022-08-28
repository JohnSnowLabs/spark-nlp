---
layout: demohomepage
title: 
full_width: true
permalink: /demosold
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
        - title: Spark NLP in Action
          subtitle: Run 300+ live demos and notebooks
      sourcetitle: Run popular demos
      source: yes
      source: 
        - title: Recognize entities in text
          id: recognize_entities_in_text
          image: 
              src: /assets/images/Split_Clean_Text_b.svg
          image2: 
              src: /assets/images/Split_Clean_Text_f.svg
          excerpt: Recognize <b>Persons, Locations, Organizations</b> and <b>Misc</b> entities using out of the box pretrained Deep Learning models based on GloVe (glove_100d) and BERT (ner_dl_bert) word embeddings.
          actions:
          - text: Live Demo
            type: normal            
            url: https://demo.johnsnowlabs.com/public/NER_EN/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_EN.ipynb
        - title: Classify documents
          id: classify_documents
          image: 
              src: /assets/images/Classifydocuments_b.svg
          image2: 
              src: /assets/images/Classify-documents-w.svg
          excerpt: Classify open-domain, fact-based questions into one of the following broad semantic categories <b>Abbreviation, Description, Entities, Human Beings, Locations or Numeric Values</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_TREC/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_TREC.ipynb
        - title: Spell check your text documents
          id: spell_check_your_text_documents
          image: 
              src: /assets/images/spelling_b.svg
          image2: 
              src: /assets/images/spelling_f.svg
          excerpt: Spark NLP contextual spellchecker allows the quick identification of typos or spell issues within any text document.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SPELL_CHECKER_EN
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb
        - title: Detect emotions in tweets
          id: detect_emotions_in_tweets
          image: 
              src: /assets/images/Detect-emotions_b.svg
          image2: 
              src: /assets/images/Detect-emotions-w.svg
          excerpt: Automatically identify <b>Joy, Surprise, Fear, Sadness</b> in Tweets using out pretrained Spark NLP DL classifier.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/public/SENTIMENT_EN_EMOTION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_EMOTION.ipynb
        - title: Recognize entities in scanned PDFs
          id: recognize_entities_in_scanned_pdfs
          image: 
              src: /assets/images/Recognize_text_in_natural_scenes_b.svg
          image2: 
              src: /assets/images/Recognize_text_in_natural_scenes_f.svg
          excerpt: 'End-to-end example of regular NER pipeline: import scanned images from cloud storage, preprocess them for improving their quality, recognize text using Spark OCR, correct the spelling mistakes for improving OCR results and finally run NER for extracting entities.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PDF_TEXT_NER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TEXT_NER.ipynb
        - title: Detect signs and symptoms
          id: detect_signs_and_symptoms
          image: 
              src: /assets/images/Detect_signs_and_symptoms_b.svg
          image2: 
              src: /assets/images/Detect_signs_and_symptoms_f.svg
          excerpt: Automatically identify <b>Signs</b> and <b>Symptoms</b> in clinical documents using two of our pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_SIGN_SYMP/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_SIGN_SYMP.ipynb
        - title: Detect temporal relations for clinical events
          id: detect_temporal_relations_for_clinical_events
          image: 
              src: /assets/images/Grammar_Analysis_b.svg
          image2: 
              src: /assets/images/Grammar_Analysis_f.svg
          excerpt: 'Automatically identify three types of relations between clinical events: After, Before and Overlap using our pretrained clinical Relation Extraction (RE) model.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_EVENTS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL_EVENTS.ipynb
        - title: De-identify PDF documents - HIPAA Compliance
          id: hipaa_compliance
          image: 
              src: /assets/images/Deidentify_PDF_documents_b.svg
          image2: 
              src: /assets/images/Deidentify_PDF_documents_f.svg
          excerpt: De-identify PDF documents using HIPAA guidelines by masking PHI information using out of the box Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_PDF_HIPAA
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_PDF.ipynb
      notebookstitle: Run Python notebooks
      notebooks: yes
      notebooks:
        - title: Spark NLP
          image: 
              src: /assets/images/notebook_1.svg
          url: https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public
        - title: Spark OCR
          image: 
              src: /assets/images/notebook_2.svg
          url: https://github.com/JohnSnowLabs/spark-ocr-workshop/tree/master/jupyter
        - title: NLU
          image: 
              src: /assets/images/notebook_3.svg
          url: https://github.com/JohnSnowLabs/nlu/tree/master/examples
        - title: Healthcare
          image: 
              src: /assets/images/notebook_4.svg
          url: https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare        
---
