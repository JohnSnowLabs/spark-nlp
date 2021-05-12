---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /extract_text_from_documents
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark OCR 
      excerpt: Extract Text from Documents
      secheader: yes
      secheader:
        - title: Spark OCR
          subtitle: Extract Text from Documents
          activemenu: extract_text_from_documents
      source: yes
      source: 
        - title: PDF to Text
          id: pdf_to_text
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
          id: dicom_to_text
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
          id: image_to_text
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
        - title: DOCX to Text
          id: docx-to-text
          image: 
              src: /assets/images/correct.svg
          image2: 
              src: /assets/images/correct_f.svg
          excerpt: Extract text from Word documents with Spark OCR
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DOCX_TO_TEXT
          - text: Colab Netbook
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOcrDocToText.ipynb
        
---
