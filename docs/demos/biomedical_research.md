---
layout: demopagenew
title: Biomedical Research  - Biomedical NLP Demos & Notebooks
seotitle: 'Biomedical NLP: Biomedical Research  - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /biomedical_research
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
        - subtitle: Biomedical Research - Live Demos & Notebooks
          activemenu: biomedical_research
      source: yes
      source:           
        - title: Detect drugs interactions
          id: detect_drugs_interactions
          image: 
              src: /assets/images/Detect_drugs_interactions.svg
          excerpt: Detect possible interactions between drugs using out-of-the-box Relation Extraction Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_DRUG_DRUG_INT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_DRUG_DRUG_INT.ipynb
        - title: PICO Classifier
          id: pico_classifier 
          image: 
              src: /assets/images/Classify-documents.svg
          excerpt: This demo shows how to classify medical texts in accordance with PICO Components.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_PICO/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_CLASSIFICATION.ipynb 
        - title: Detect relations between chemicals and proteins
          id: detect_relations_between_chemicals_and_proteins
          image: 
              src: /assets/images/Detect_relations_between_chemicals_and_proteins.svg
          excerpt: Automatically detect possible relationships between chemicals and proteins using a predefined Relation Extraction model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_CHEM_PROT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_CHEM_PROT.ipynb
        - title: Extract relations between drugs and proteins 
          id: extract_relations_between_drugs_proteins 
          image: 
              src: /assets/images/Extract_relations_between_drugs_and_proteins.svg
          excerpt: This model detects interactions between chemical compounds/drugs and genes/proteins.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/RE_DRUG_PROT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_DRUG_PROT.ipynb 
        - title: Detect Pathogen Concepts  
          id: detect_pathogen_concepts    
          image: 
              src: /assets/images/DetectPathogenConcepts.svg
          excerpt: This demo automatically identifies pathogen concepts from clinical text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_PATHOGEN/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_PATHOGEN.ipynb
        - title: Detect mentions of general medical terms (coarse) 
          id: detect_mentions_general_medical_terms_coarse
          image: 
              src: /assets/images/Detect_mentions_of_general_medical_terms.svg
          excerpt: Extract general medical terms in text like body parts, cells, genes, symptoms, etc in text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_MEDMENTIONS_COARSE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_MEDMENTIONS_COARSE.ipynb
---