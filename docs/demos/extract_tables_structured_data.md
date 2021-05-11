---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /extract_tables_structured_data
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark OCR 
      excerpt: Extract Tables & Structured Data
      secheader: yes
      secheader:
        - title: Spark OCR
          subtitle: Extract Tables & Structured Data
          activemenu: extract_tables_structured_data
      source: yes
      source: 
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
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TEXT_NER.ipynb
        - title: Extract tables
          id: extract_tables_from_pdfs
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
        - title: Extract Data from FoundationOne Sequencing Reports
          id: extract-data-from-foundationone-sequencing-reports
          image: 
              src: /assets/images/correct.svg
          image2: 
              src: /assets/images/correct_f.svg
          excerpt: Use our transformer to parse patient info, genomic and biomarker findings, and gene lists.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/FOUNDATIONONE_REPORT_PARSING/
          - text: Colab Netbook
            type: blue_btn
            url:      
---
