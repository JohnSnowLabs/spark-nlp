---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /resolve_entities_codes
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Resolve Entities to Codes 
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Resolve Entities to Codes 
          activemenu: resolve_entities_codes
      source: yes
      source: 
        - title: SNOMED coding
          id: snomed_coding
          image: 
              src: /assets/images/Detect_signs_and_symptoms.svg
          image2: 
              src: /assets/images/Detect_signs_and_symptoms_f.svg
          excerpt: Automatically resolve the SNOMED code corresponding to the diseases and conditions mentioned in your health record using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_SNOMED
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_SNOMED.ipynb
        - title: ICDO coding
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
        - title: ICD10-CM coding 
          id: icd10-cm_coding
          image: 
              src: /assets/images/Detect_risk_factors.svg
          image2: 
              src: /assets/images/Detect_risk_factors_f.svg
          excerpt: Automatically detect the pre and post op diagnosis, signs and symptoms or other findings in your healthcare records and automatically link them to the corresponding ICD10-CM code using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb
        - title: RxNORM coding
          id: rxnorm_coding
          image: 
              src: /assets/images/Detect_drugs_and_prescriptions.svg
          image2: 
              src: /assets/images/Detect_drugs_and_prescriptions_f.svg
          excerpt: Automatically detect the drugs and treatments names mentioned in your prescription or healthcare records and link them to the corresponding RxNORM codes using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_RXNORM
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_RXNORM.ipynb               
---
