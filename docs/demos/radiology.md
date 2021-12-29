---
layout: demopage
title: Radiology
full_width: true
permalink: /radiology
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare
      excerpt: Radiology
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Radiology
          activemenu: radiology
      source: yes
      source: 
        - title: Detect Clinical Entities in Radiology Reports
          id: detect_clinical_entities_in_radiology_reports
          image: 
              src: /assets/images/Detect_Clinical_Entities_in_Radiology_Reports.svg
          image2: 
              src: /assets/images/Detect_Clinical_Entities_in_Radiology_Reports_f.svg
          excerpt: Automatically identify entities such as body parts, imaging tests, imaging results and diseases using a pre-trained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_RADIOLOGY
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb
        - title: Detect Anatomical and Observation Entities in Chest Radiology Reports
          id: detect_anatomical_observation_entities_chest_radiology_reports 
          image: 
              src: /assets/images/Detect_Anatomical_and_Observation_Entities_in_Chest_Radiology_Reports.svg
          image2: 
              src: /assets/images/Detect_Anatomical_and_Observation_Entities_in_Chest_Radiology_Reports_f.svg
          excerpt: This demo shows how Anatomical and Observation entities can be extracted from Chest Radiology Reports.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEXPERT/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb
        
---