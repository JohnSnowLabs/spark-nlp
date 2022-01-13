---
layout: demopage
title: Oncology
full_width: true
permalink: /oncology
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare
      excerpt: Oncology
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Oncology
          activemenu: oncology
      source: yes
      source: 
        - title: Detect oncological & biological concepts
          id: detect_tumor_characteristics
          image: 
              src: /assets/images/Detect_tumor_characteristics.svg
          image2: 
              src: /assets/images/Detect_tumor_characteristics_f.svg
          excerpt: Automatically identify <b>oncological</b> and <b>biological</b> entities such as <b>Amino_acids, Anatomical systems, Cancer, Cells or Cellular components</b> using our pertained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_TUMOR
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR.ipynb  
        - title: Map oncology terminology to ICD-O taxonomy
          id: icdo_coding
          image: 
              src: /assets/images/Detect_diagnosis_and_procedures.svg
          image2: 
              src: /assets/images/Detect_diagnosis_and_procedures_f.svg
          excerpt: Automatically detect the tumor in your healthcare records and link it to the corresponding ICDO code using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICDO
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICDO.ipynb
---