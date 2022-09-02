---
layout: demopagenew
title: Extract Financial Relationships - Finance NLP Demos & Notebooks
seotitle: 'Finance NLP: Extract Financial Relationships - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /financial_relation_extraction
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
        - subtitle: Extract Financial Relationships - Live Demos & Notebooks
          activemenu: financial_relation_extraction
      source: yes
      source: 
        - title: Extract Relations between Organizations, Products and their Aliases  
          id: extract_relations_between_orgs_prods_aliases 
          image: 
              src: /assets/images/Extract_Relations_between_Parties.svg
          image2: 
              src: /assets/images/Extract_Relations_between_Parties_f.svg
          excerpt: This model uses Entity Recognition to identify ORG (Companies), PRODUCT (Products) and their ALIAS in financial documents. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINRE_ALIAS/
          - text: Colab
            type: blue_btn
            url: 
        - title: Extract Acquisition and Subsidiary Relationships  
          id: extract_acquisition_subsidiary_relationships  
          image: 
              src: /assets/images/Extract_Acquisition_and_Subsidiary_Relationships.svg
          image2: 
              src: /assets/images/Extract_Acquisition_and_Subsidiary_Relationships_f.svg
          excerpt: This demo shows how to extract Acquisition and Subsidiary relations from ORG (Companies), ALIAS (Aliases of companies in an agreement) and PRODUCT (Products).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINRE_ACQUISITIONS/
          - text: Colab
            type: blue_btn
            url: https://nlp.johnsnowlabs.com/  
        - title: Financial Zero-shot Relation Extraction   
          id: financial_zero_shot_relation_extraction   
          image: 
              src: /assets/images/Financial_Zero_shot_Relation_Extraction.svg
          image2: 
              src: /assets/images/Financial_Zero_shot_Relation_Extraction_f.svg
          excerpt: This demo shows how you can carry out Relation Extraction without training any model, just with some textual examples.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/FINRE_ZEROSHOT/
          - text: Colab
            type: blue_btn
            url:       
---