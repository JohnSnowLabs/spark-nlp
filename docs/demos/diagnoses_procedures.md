---
layout: demopagenew
title: Diagnoses & Procedures - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Diagnoses & Procedures - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /diagnoses_procedures
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
        - subtitle: Diagnoses & Procedures - Live Demos & Notebooks
          activemenu: diagnoses_procedures
      source: yes
      source:           
        - title: Identify diagnosis and symptoms assertion status
          id: identify_diagnosis_and_symptoms_assertion_status
          image: 
              src: /assets/images/Identify_diagnosis_and_symptoms_assertion_status.svg
          excerpt: Automatically detect if a diagnosis or a symptom is present, absent, uncertain or associated to other persons (e.g. family members).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ASSERTION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb
        - title: Calculate Medicare HCC Risk Score
          id: calculate_medicare_risk_score 
          image: 
              src: /assets/images/Calculate_Medicare_Risk_Score.svg
          excerpt: This demos shows how to calculate medical risk adjustment scores automatically using ICD codes of diseases.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/HCC_RISK_SCORE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb
        - title: Detect clinical entities in text
          id: detect_clinical_entities_in_text
          image: 
              src: /assets/images/Detect_risk_factors.svg
          excerpt: Automatically detect more than 50 clinical entities using our NER deep learning model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CLINICAL/
          - text: Colab
            type: blue_btn
            url: https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb
        - title: Detect diagnosis and procedures
          id: detect_diagnosis_and_procedures
          image: 
              src: /assets/images/Detect_diagnosis_and_procedures.svg
          excerpt: Automatically identify diagnoses and procedures in clinical documents using the pretrained Spark NLP clinical models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DIAG_PROC/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DIAG_PROC.ipynb
        - title: Detect temporal relations for clinical events
          id: detect_temporal_relations_for_clinical_events
          image: 
              src: /assets/images/Grammar_Analysis.svg
          excerpt: 'Automatically identify three types of relations between clinical events: After, Before and Overlap using our pretrained clinical Relation Extraction (RE) model.'
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_EVENTS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL_EVENTS.ipynb
        - title: Detect causality between symptoms and treatment
          id: detect_causality_between_symptoms_and_treatment
          image: 
              src: /assets/images/Grammar_Analysis.svg
          excerpt: Automatically identify relations between symptoms and treatment using our pretrained clinical Relation Extraction (RE) model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL.ipynb
        - title: Detect relations between body parts and clinical entities
          id: detect_relations_between_body_parts_and_clinical_entities
          image: 
              src: /assets/images/Detect_relations.svg
          excerpt: Use pre-trained relation extraction models to extract relations between body parts and clinical entities.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_BODYPART_ENT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb
        - title: Detect how dates relate to clinical entities
          id: detect_how_dates_relate_to_clinical_entities
          image: 
              src: /assets/images/ExtractRelationships_2.svg
          excerpt: Detect clinical entities such as problems, tests and treatments, and how they relate to specific dates.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL_DATE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CLINICAL_DATE.ipynb
        - title: Detecting Clinical Entities with JSL
          id: detecting_clinical_entities_jsl
          image: 
              src: /assets/images/Detect_Clinical_Entities_with_JSL.svg
          excerpt: Automatically detects more than 77 clinical entities using our NER deep learning model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_JSL/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb
---