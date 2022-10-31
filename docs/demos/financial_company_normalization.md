---
layout: demopagenew
title: Normalization & Data Augmentation - Finance NLP Demos & Notebooks
seotitle: 'Finance NLP: Normalization & Data Augmentation - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /financial_company_normalization
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
          activemenu: financial_company_normalization
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
          id: augment_company_names_public_information   
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
        - title: Financial Graph Visualization 
          id: financial_graph_visualization   
          image: 
              src: /assets/images/Financial_Graph_Visualization.svg
          excerpt: Use different models from Spark NLP for Finance, as NER, Relation Extraction, Entity Resolution and Chunk Mappers, to create your own Financial Graphs.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/NEO4J/
          - text: Colab
            type: blue_btn
            url:        
---