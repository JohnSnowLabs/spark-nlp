---
layout: demopagenew
title: Radiology - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Radiology - John Snow Labs'
full_width: true
permalink: /radiology
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
        - subtitle: Radiology - Live Demos & Notebooks
          activemenu: radiology
      source: yes
      source: 
        - title: Detect Clinical Entities in Radiology Reports
          id: detect_clinical_entities_in_radiology_reports
          image: 
              src: /assets/images/Detect_Clinical_Entities_in_Radiology_Reports.svg
          excerpt: Automatically identify entities such as body parts, imaging tests, imaging results and diseases using a pre-trained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_RADIOLOGY
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_RADIOLOGY.ipynb
        - title: Detect Anatomical and Observation Entities in Chest Radiology Reports
          id: detect_anatomical_observation_entities_chest_radiology_reports 
          image: 
              src: /assets/images/Detect_Anatomical_and_Observation_Entities_in_Chest_Radiology_Reports.svg
          excerpt: This demo shows how Anatomical and Observation entities can be extracted from Chest Radiology Reports.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEXPERT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEXPERT.ipynb
        - title: Assign an assertion status (confirmed, suspected or negative) to Image Findings
          id: identify_assertion_status_image_findings_Radiology  
          image: 
              src: /assets/images/Identify_assertion_status_for_image_findings_of_Radiology.svg
          excerpt: This demo shows how Imaging-Findings in Radiology reports can be detected as confirmed, suspected or negative using a Spark NLP Assertion Status model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ASSERTION_RADIOLOGY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ASSERTION_RADIOLOGY.ipynb
        - title: Identify relations between problems, tests and findings
          id: identify_relations_between_problems_tests_findings  
          image: 
              src: /assets/images/Identify_relations_between_problems_tests_and_findings.svg
          excerpt: This demo shows how relations between problems, tests and findings in radiology reports can be identified using a Spark NLP RE model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_RADIOLOGY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_RADIOLOGY.ipynb
---