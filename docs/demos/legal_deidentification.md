---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /legal_deidentification
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Legal
      excerpt: Legal De-Identification
      secheader: yes
      secheader:
        - title: Spark NLP for Legal
          subtitle: Legal De-Identification
          activemenu: legal_deidentification
      source: yes
      source: 
        - title: Deidentify structured data
          id: deidentify_structured_data
          image: 
              src: /assets/images/Deidentify_structured_data.svg
          image2: 
              src: /assets/images/Deidentify_structured_data_f.svg
          excerpt: Deidentify PHI information from structured datasets using out of the box Spark NLP functionality that enforces GDPR and HIPPA compliance, while maintaining linkage of clinical data across files.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DEID_EHR_DATA
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_EHR_DATA.ipynb
        - title: Deidentify free text documents
          id: deidentify_free_text_documents
          image: 
              src: /assets/images/Deidentify_free_text_documents.svg
          image2: 
              src: /assets/images/Deidentify_free_text_documents_f.svg
          excerpt: Deidentify free text documents by either masking or obfuscating PHI information using out of the box Spark NLP models that enforce GDPR and HIPPA compliance.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT.ipynb
        - title: Deidentify DICOM documents
          id: deidentify_dicom_documents
          image: 
              src: /assets/images/Deidentify_DICOM_documents.svg
          image2: 
              src: /assets/images/Deidentify_DICOM_documents_f.svg
          excerpt: Deidentify DICOM documents by masking PHI information on the image and by either masking or obfuscating PHI from the metadata.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_DICOM_IMAGE
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_DICOM_IMAGE.ipynb
        - title: De-identify PDF documents - HIPAA Compliance
          id: hipaa_compliance
          image: 
              src: /assets/images/Deidentify_PDF_documents.svg
          image2: 
              src: /assets/images/Deidentify_PDF_documents_f.svg
          excerpt: De-identify PDF documents using HIPAA guidelines by masking PHI information using out of the box Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_PDF_HIPAA
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_PDF.ipynb
        - title: De-identify PDF documents - GDPR Compliance
          id: gdpr_compliance
          image: 
              src: /assets/images/Deidentify_PDF_documents.svg
          image2: 
              src: /assets/images/Deidentify_PDF_documents_f.svg
          excerpt: De-identify PDF documents using GDPR guidelines by anonymizing PHI information using out of the box Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/ocr/DEID_PDF_GDPR
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/ocr/DEID_PDF.ipynb   
        - title: Detect PHI Entities for Deidentification
          id: detect_phi_entities
          image: 
              src: /assets/images/Detect_PHI_Entities.svg
          image2: 
              src: /assets/images/Detect_PHI_Entities_f.svg
          excerpt: This demo shows how Protected Health Information (PHI) that may need to be de-identified can be extracted using Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb
        - title: Deidentify Clinical Notes in Different Languages
          id: deidentify_clinical_notes_different_languages
          image: 
              src: /assets/images/Deidentify_free_text_documents.svg
          image2: 
              src: /assets/images/Deidentify_free_text_documents_f.svg
          excerpt: This demo shows how to deidentify protected health information in English , Spanish, French, Italian and German texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT_MULTI.ipynb
---
