---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /financial_company_normalization
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Finance
      excerpt: Normalize and Augment Company Names
      secheader: yes
      secheader:
        - title: Spark NLP for Finance
          subtitle: Normaliz and Augment Company Names
          activemenu: financial_company_normalization
      source: yes
      source: 
        - title: Company Normalization for Edgar and Crunchbase databases 
          id: company_normalization_edgar_crunchbase_databases 
          image: 
              src: /assets/images/Company_Normalization.svg
          image2: 
              src: /assets/images/Company_Normalization_f.svg
          excerpt: These models normalize versions of Company Names using Edgar and Crunchbase databases conventions.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/ER_EDGAR_CRUNCHBASE/
          - text: Colab Netbook
            type: blue_btn
            url:    
        - title: Augment Company Names with Public Information  
          id: augment_company_names_public_information   
          image: 
              src: /assets/images/Augment_Company_Names_Public_Information.svg
          image2: 
              src: /assets/images/Augment_Company_Names_Public_Information_f.svg
          excerpt: These models aim to augment NER with information from external sources.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FIN_LEG_COMPANY_AUGMENTATION 
          - text: Colab Netbook
            type: blue_btn
            url:              
---