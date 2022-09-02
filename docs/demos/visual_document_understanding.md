---
layout: demopagenew
title: Visual Document Understanding - Visual NLP Demos & Notebooks
seotitle: 'Visual NLP: Visual Document Understanding - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /visual_document_understanding
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
        - subtitle: Visual Document Understanding - Live Demos & Notebooks
          activemenu: visual_document_understanding
      source: yes
      source: 
        - title: Visual Document Classification
          id: classify_visual_documents
          image: 
              src: /assets/images/Classify_visual_documents.svg
          image2: 
              src: /assets/images/Classify_visual_documents_f.svg
          excerpt: Classify documents using text and layout data with the new features offered by Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/VISUAL_DOCUMENT_CLASSIFY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRVisualDocumentClassifier.ipynb
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
        - title: Extract Data from FoundationOne Sequencing Reports
          id: extract-data-from-foundationone-sequencing-reports
          image: 
              src: /assets/images/correct.svg
          image2: 
              src: /assets/images/correct_f.svg
          excerpt: Extract patient, genomic and biomarker information from FoundationOne Sequencing Reports.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/FOUNDATIONONE_REPORT_PARSING/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/FOUNDATIONONE_REPORT_PARSING.ipynb 
        - title: Recognize entities in scanned PDFs
          id: recognize_entities_in_scanned_pdfs
          image: 
              src: /assets/images/Recognize_text_in_natural_scenes.svg
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
        - title: Extract brands from visual documents
          id: extract_brands_from_visual_documents 
          image: 
              src: /assets/images/Extract_brands_from_visual_documents.svg
          image2: 
              src: /assets/images/Extract_brands_from_visual_documents_f.svg
          excerpt: This demo shows how brands from image can be detected using Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/BRAND_EXTRACTION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/BRAND_EXTRACTION.ipynb
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
        - title: Classify Financial Documents using SparkOCR 
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
---
