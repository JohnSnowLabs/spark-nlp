---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /extract_handwritten_texts
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark OCR
      excerpt: Extract handwritten texts
      secheader: yes
      secheader:
        - title: Spark OCR
          subtitle: Extract handwritten texts
          activemenu: extract_handwritten_texts
      source: yes
      source: 
        - title: Extract Signatures
          id: extract_signatures 
          image: 
              src: /assets/images/Extract_Signatures.svg
          image2: 
              src: /assets/images/Extract_Signatures_c.svg
          excerpt: This demo shows how handwritten signatures can be extracted from image/pdf documents using Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DETECT_SIGNATURES/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/3.6.0/jupyter/SparkOcrImageSignatureDetection.ipynb
        
---
