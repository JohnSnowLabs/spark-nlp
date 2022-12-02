---
layout: demopagenew
title: Normalization & Data Augmentation - Legal NLP Demos & Notebooks
seotitle: 'Legal NLP: Normalization & Data Augmentation - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /legal_company_normalization
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
        - subtitle: Normalization & Data Augmentation - Live Demos & Notebooks
          activemenu: legal_company_normalization
      source: yes
      source: 
        - title: Company names Normalization 
          id: company_normalization_edgar_crunchbase_databases 
          image: 
              src: /assets/images/Company_Normalization.svg
          excerpt: These models normalize versions of Company Names using Edgar and Crunchbase databases conventions.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/ER_EDGAR_CRUNCHBASE/
          - text: Colab
            type: blue_btn
            url:   
        - title: Augment Company Names with Public Information  
          id: augment_company_names_public_information_legal  
          image: 
              src: /assets/images/Augment_Company_Names_Public_Information.svg
          excerpt: These models aim to augment NER with information from external sources.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION 
          - text: Colab
            type: blue_btn
            url:              
---