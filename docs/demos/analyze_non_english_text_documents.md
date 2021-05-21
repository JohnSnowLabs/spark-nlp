---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /analyze_non_english_text_documents
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark OCR 
      excerpt: Analyze Non-English Text & Documents
      secheader: yes
      secheader:
        - title: Spark OCR
          subtitle: Analyze Non-English Text & Documents
          activemenu: analyze_non_english_text_documents
      source: yes
      source: 
        - title: PDF to Text (Non-English Text)
          id: pdf-to-text-non-english-text
          image: 
              src: /assets/images/PDF_to_Text.svg
          image2: 
              src: /assets/images/PDF_to_Text_f.svg
          excerpt: Extract non-English text from generated/selectable PDF documents and keep the original structure of the document by using our out-of-the-box Spark OCR library.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/PDF_TO_TEXT_MUL/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/PDF_TO_TEXT.ipynb
        - title: Image to Text (Non-English Text)
          id: image-to-text-non-english-text
          image: 
              src: /assets/images/Image_to_Text.svg
          image2: 
              src: /assets/images/Image_to_Text_f.svg
          excerpt: Recognize non-English text in images and scanned PDF documents by using our out-of-the-box Spark OCR library.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/IMAGE_TO_TEXT_MUL/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/IMAGE_TO_TEXT.ipynb
        - title: DOCX to Text (Non-English Text)
          id: docx-to-text-non-english-text
          image: 
              src: /assets/images/correct.svg
          image2: 
              src: /assets/images/correct_f.svg
          excerpt: Extract non-English text from Word documents using out out-of-the-box Spark OCR library.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DOCX_TO_TEXT_MUL/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DOCX_TO_TEXT.ipynb
        
---
