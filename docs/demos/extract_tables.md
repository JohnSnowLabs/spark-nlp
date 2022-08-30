---
layout: demopagenew
title: Extract Tables - Visual NLP Demos & Notebooks
seotitle: 'Visual NLP: Extract Tables - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /extract_tables
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
        - subtitle: Extract Tables - Live Demos & Notebooks
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
          - text: Colab
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
          - text: Colab
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
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrImageTableDetection.ipynb
        - title: Extract tables from Powerpoint slides 
          id: extract_tables_from_power_point_slide  
          image: 
              src: /assets/images/PPTX_to_Table.svg
          image2: 
              src: /assets/images/PPTX_to_Table_f.svg
          excerpt: This demo shows how PPTX tables can be extracted using Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PPTX_TABLE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PPTX_TABLE.ipynb
---
