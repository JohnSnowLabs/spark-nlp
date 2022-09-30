---
layout: demopagenew
title: Financial Document Understanding - Finance NLP Demos & Notebooks
seotitle: 'Visual NLP: Financial Document Understanding - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /financial_document_understanding
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
        - subtitle: Financial Document Understanding - Live Demos & Notebooks
          activemenu: financial_document_understanding
      source: yes
      source: 
        - title: Classify Financial Documents 
          id: classify_financial_documents_using_spark_ocr 
          image: 
              src: /assets/images/Classify_Financial_Documents_using_SparkOCR.svg
          image2: 
              src: /assets/images/Classify_Financial_Documents_using_SparkOCR_f.svg
          excerpt: This demo shows how to classify finance documents using text and layout data with the new features offered by Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/VISUAL_DOCUMENT_CLASSIFICATION_V3/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/VISUAL_DOCUMENT_CLASSIFICATION_V3.ipynb
        - title: Extract Data from Scanned Invoices
          id: extract_entities_from_visual_documents  
          image: 
              src: /assets/images/Extract_entities_from_visual_documents.svg
          image2: 
              src: /assets/images/Extract_entities_from_visual_documents_c.svg
          excerpt: Detect companies, total amounts and dates in scanned invoices using out of the box Spark OCR models. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/VISUAL_DOCUMENT_NER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRVisualDocumentNer.ipynb
        - title: Form Recognition
          id: form_recognition 
          image: 
              src: /assets/images/Detect_sentences_in_text.svg
          image2: 
              src: /assets/images/Detect_sentences_in_text_f.svg
          excerpt: This demo shows how to perceive data in forms as key-value pairs.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/FORM_RECOGNITION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/FormRecognition/SparkOcrFormRecognition.ipynb                
---