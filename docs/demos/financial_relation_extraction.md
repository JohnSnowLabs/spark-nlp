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
---