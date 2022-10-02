---
layout: demopagenew
title: Extract Legal Relationships - Legal NLP Demos & Notebooks
seotitle: 'Legal NLP: Extract Legal Relationships - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /legal_relation_extraction
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
        - subtitle: Extract Legal Relationships - Live Demos & Notebooks
          activemenu: legal_relation_extraction
      source: yes
      source: 
        - title: Legal Zero-shot Relation Extraction  
          id: legal_zero_shot_relation_extraction   
          image: 
              src: /assets/images/Legal_Zero-shot_Relation_Extraction.svg
          image2: 
              src: /assets/images/Legal_Zero-shot_Relation_Extraction_f.svg
          excerpt: This demo shows how you can carry out Relation Extraction without training any model, just with some textual examples.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGRE_ZEROSHOT/
          - text: Colab
            type: blue_btn
            url: 
        - title: Extract Relations between Parties in agreements  
          id: extract_relations_between_parties_agreement  
          image: 
              src: /assets/images/Extract_Relations_between_Parties.svg
          image2: 
              src: /assets/images/Extract_Relations_between_Parties_f.svg
          excerpt: This model uses Deep Learning Name Entity Recognition and a Relation Extraction models to extract the document type (DOC), the Effective Date (EFFDATE), the PARTIES in an agreement and their ALIAS (separate and collectively).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGALRE_PARTIES/
          - text: Colab
            type: blue_btn
            url:
        - title: Extract Syntactic Relationships in Legal sentences 
          id: extract_syntactic_relationships_legal_sentences    
          image: 
              src: /assets/images/Extract_Syntactic_Relationships_in_Legal_sentences.svg
          image2: 
              src: /assets/images/Extract_Syntactic_Relationships_in_Legal_sentences_f.svg
          excerpt: This demo shows how legal sentence elements can be accessed using syntactic relationships (dependency parser).
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/legal/LEGPIPE_RE/
          - text: Colab
            type: blue_btn
            url:             
---