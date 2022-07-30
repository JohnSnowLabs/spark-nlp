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
        - title: Extract Relations between Parties in an Agreement  
          id: extract_relations_between_parties_agreement  
          image: 
              src: /assets/images/Extract_Relations_between_Parties.svg
          image2: 
              src: /assets/images/Extract_Relations_between_Parties_f.svg
          excerpt: This model uses Deep Learning Name Entity Recognition and a Relation Extraction models to extract the document type (DOC), the Effective Date (EFFDATE), the PARTIES in an agreement and their ALIAS (separate and collectively).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/finance/LEGALRE_PARTIES/
          - text: Colab Netbook
            type: blue_btn
            url:                 
---