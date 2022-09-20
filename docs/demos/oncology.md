---
layout: demopagenew
title: Oncology - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Oncology - John Snow Labs'
full_width: true
permalink: /oncology
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
        - subtitle: Oncology - Live Demos & Notebooks
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
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR.ipynb  
        - title: Resolve Oncology terminology using the ICD-O taxonomy
          id: resolve_oncology_terminology_using_icdo_taxonomy
          image: 
              src: /assets/images/Resolve_Oncology_terminology.svg
          image2: 
              src: /assets/images/Resolve_Oncology_terminology_f.svg
          excerpt: This model maps oncology terminology to ICD-O codes using Entity Resolvers.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICDO/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICDO.ipynb
        - title: Extract the biomarker information
          id: extract_biomarker_information 
          image: 
              src: /assets/images/Extract_brands_from_visual_documents.svg
          image2: 
              src: /assets/images/Extract_brands_from_visual_documents_f.svg
          excerpt: This demo shows how biomarkers, therapies, oncological, and other general concepts can be extracted using Spark NLP Healthcare NER model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_BIOMARKER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BIOMARKER.ipynb  
        - title: Detect Oncological Concepts
          id: detect_oncological_concepts   
          image: 
              src: /assets/images/Detect_Oncological_Concepts.svg
          excerpt: Automatically identify oncological concepts from clinical text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb
---