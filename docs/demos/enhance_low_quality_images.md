---
layout: demopagenew
title: Enhance Low-Quality Images - Visual NLP Demos & Notebooks
seotitle: 'Visual NLP: Enhance Low-Quality Images - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /enhance_low_quality_images
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
        - subtitle: Enhance Low-Quality Images - Live Demos & Notebooks
          activemenu: enhance_low_quality_images
      source: yes
      source: 
        - title: Remove background noise from scanned documents
          id: remove_background_noise_from_scanned_documents
          image: 
              src: /assets/images/remove_bg.svg
          excerpt: Removing the background noise in a scanned document will highly improve the results of the OCR. Spark OCR is the only library that allows you to finetune the image preprocessing for excellent OCR results.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/BG_NOISE_REMOVER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/BG_NOISE_REMOVER.ipynb
        - title: Correct skewness in scanned documents
          id: correct_skewness_in_scanned_documents
          image: 
              src: /assets/images/correct.svg
          excerpt: Correct the skewness of your scanned documents will highly improve the results of the OCR. Spark OCR is the only library that allows you to finetune the image preprocessing for excellent OCR results.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/SKEW_CORRECTION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/SKEW_CORRECTION.ipynb
        - title: Recognize text in natural scenes
          id: recognize_text_in_natural_scenes
          image: 
              src: /assets/images/Frame.svg
          excerpt: By using image segmentation and preprocessing techniques Spark OCR recognizes and extracts text from natural scenes.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/NATURAL_SCENE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/NATURAL_SCENE.ipynb
        - title: Enhance Faxes or Scanned Documents
          id: enhance_faxes_scanned_documents
          image: 
              src: /assets/images/Healthcare_Enhancelowquality.svg
          excerpt: Improve quality of (old) faxes/scanned documents using Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/ENHANCE_OLD_FAXES/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRGPUOperations.ipynb
        - title: Enhance Photo of Documents
          id: enhance_photo_documents
          image: 
              src: /assets/images/Healthcare_EnhancePhotoofDocuments.svg
          excerpt: Improve quality of documents in image format using Spark OCR.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/ENHANCE_DOC_PHOTO/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-ocr-workshop/blob/master/jupyter/SparkOCRGPUOperations.ipynb
        
---
