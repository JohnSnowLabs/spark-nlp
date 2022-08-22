---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /financial_relation_extraction
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Finance
      excerpt: Financial Relation Extraction
      secheader: yes
      secheader:
        - title: Spark NLP for Finance
          subtitle: Financial Relation Extraction
          activemenu: financial_relation_extraction
      source: yes
      source: 
        - title: Extract Relations between ORGS/PRODS and their ALIASES  
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
          - text: Colab Netbook
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
          - text: Colab Netbook
            type: blue_btn
            url: https://nlp.johnsnowlabs.com/             
---