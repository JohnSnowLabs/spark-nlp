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
        - title: Normalize & Augment Company Information with Wikidata
          id: normalize_augment_company_information_wikidata    
          image: 
              src: /assets/images/Normalize_Augment_Company_Information_with_Wikidata.svg
          excerpt: This demo shows how to apply NER or Assertion Status to texts from Wikipedia. In addition, shows how you can create data dumps from Wikidata to include them in Spark NLP and use them online for data augmentation purposes, using Chunk Mappers and Entity Resolution.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINANCE_NLP_WITH_WIKIDATA/
          - text: Colab
            type: blue_btn
            url:    
---