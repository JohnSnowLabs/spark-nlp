---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /extract_tables
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark OCR 
      excerpt: Extract Tables
      secheader: yes
      secheader:
        - title: Spark OCR
          subtitle: Extract Tables
          activemenu: extract_tables
      source: yes
      source: 
        - title: Extract tables from selectable PDFs
          id: extract_tables_from_pdfs
          image: 
              src: /assets/images/Extract_tables_from_PDFs.svg
          image2: 
              src: /assets/images/Extract_tables_from_PDFs_f.svg
          excerpt: Extract tables from selectable PDF documents with Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PDF_TEXT_TABLE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TEXT_TABLE.ipynb
        - title: Detect tables in scanned documents
          hide: yes
          id: detect_tables_in_documents
          image: 
              src: /assets/images/Detect_tables_in_documents.svg
          image2: 
              src: /assets/images/Detect_tables_in_documents_f.svg
          excerpt: Detect tables on the image by a pretrained model based on CascadeTabNet.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/TABLE_DETECTION/ 
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrImageTableDetection.ipynb
        - title: Detect and extract tables in scanned PDFs 
          id: detect_tables_extract_text 
          image: 
              src: /assets/images/Detect_sentences_in_text.svg
          image2: 
              src: /assets/images/Detect_sentences_in_text_f.svg
          excerpt: Detect and extract structured tables from scanned PDF documents & images with Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/IMAGE_TABLE_DETECTION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrImageTableDetection.ipynb
---
