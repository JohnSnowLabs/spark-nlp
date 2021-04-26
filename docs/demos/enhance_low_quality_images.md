---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /enhance_low_quality_images
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark OCR 
      excerpt: Enhance Low-Quality Images
      secheader: yes
      secheader:
        - title: Spark OCR
          subtitle: Enhance Low-Quality Images
          activemenu: enhance_low_quality_images
      source: yes
      source: 
        - title: Remove background noise from scanned documents
          id: remove_background_noise_from_scanned_documents
          image: 
              src: /assets/images/remove_bg.svg
          image2: 
              src: /assets/images/remove_bg_f.svg
          excerpt: Removing the background noise in a scanned document will highly improve the results of the OCR. Spark OCR is the only library that allows you to finetune the image preprocessing for excellent OCR results.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/BG_NOISE_REMOVER/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/BG_NOISE_REMOVER.ipynb
        - title: Correct skewness in scanned documents
          id: correct_skewness_in_scanned_documents
          image: 
              src: /assets/images/correct.svg
          image2: 
              src: /assets/images/correct_f.svg
          excerpt: Correct the skewness of your scanned documents will highly improve the results of the OCR. Spark OCR is the only library that allows you to finetune the image preprocessing for excellent OCR results.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/SKEW_CORRECTION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/SKEW_CORRECTION.ipynb
        - title: Recognize text in natural scenes
          id: recognize_text_in_natural_scenes
          image: 
              src: /assets/images/Frame.svg
          image2: 
              src: /assets/images/Frame_f.svg
          excerpt: By using image segmentation and preprocessing techniques Spark OCR recognizes and extracts text from natural scenes.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/NATURAL_SCENE/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/NATURAL_SCENE.ipynb
        
---
