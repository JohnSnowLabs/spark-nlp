---
layout: demopagenew
title: Analyze Clinical Notes - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Analyze Clinical Notes - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /analyze_clinical_notes
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
        - subtitle: Analyze Clinical Notes - Live Demos & Notebooks
          activemenu: analyze_clinical_notes
      source: yes
      source:           
        - title: Normalize Section Headers of the Visit Summary 
          id: normalize_section_headers_visit_summary 
          image: 
              src: /assets/images/Normalize_Section_Headers_of_the_Visit_Summary.svg
          excerpt: This demo maps Section Headers of the clinical visit data to their normalized versions.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NORMALIZED_SECTION_HEADER_MAPPER/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NORMALIZED_SECTION_HEADER_MAPPER.ipynb
        - title: Resolve Clinical Abbreviations and Acronyms
          id: resolve_clinical_abbreviations_acronyms    
          image: 
              src: /assets/images/Resolve_Clinical_Abbreviations_and_Acronyms.svg
          excerpt: This demo shows how to map clinical abbreviations and acronyms to their meanings.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_CLINICAL_ABBREVIATION_ACRONYM/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_CLINICAL_ABBREVIATION_ACRONYM.ipynb
        - title: Spell checking for clinical documents
          id: spell_checking_for_clinical_documents
          image: 
              src: /assets/images/Detect_clinical_events.svg
          excerpt: Automatically identify from clinical documents using our pretrained Spark NLP model <b>ner_bionlp.</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_SPELL_CHECKER.ipynb
        - title: Detect sentences in healthcare documents
          id: detect_sentences_in_healthcare_documents
          image: 
              src: /assets/images/Detect_sentences_in_healthcare_documents.svg
          excerpt: Automatically detect sentences in noisy healthcare documents with our pretrained Sentence Splitter DL model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/SENTENCE_DETECTOR_HC/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb
        - title: Find available models for your clinical entities 
          id: ner_model_finder
          image: 
              src: /assets/images/NER_Model_Finder.svg
          excerpt: This demo shows how to use a pretrained pipeline to find the best NER model given an entity name.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_MODEL_FINDER/
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb
        - title: Normalize medication-related phrases
          id: normalize_medication-related_phrases
          image: 
              src: /assets/images/Normalize_Medication-related_Phrases.svg
          excerpt: Normalize medication-related phrases such as dosage, form and strength, as well as abbreviations in text and named entities extracted by NER models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DRUG_NORMALIZATION
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/23.Drug_Normalizer.ipynb
        - title: Detect anatomical references
          id: detect_anatomical_references
          image: 
              src: /assets/images/Detect_anatomical_references.svg
          excerpt: Automatically identify <b>Anatomical System, Cell, Cellular Component, Anatomical Structure, Immaterial Anatomical Entity, Multi-tissue Structure, Organ, Organism Subdivision, Organism Substance, Pathological Formation</b> in clinical documents using our pretrained Spark NLP model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_ANATOMY.ipynb
        - title: Extract Chunk Key Phrases 
          id: extract_chunk_key_phrases  
          image: 
              src: /assets/images/Extract_Chunk_Key_Phrases.svg
          excerpt: This demo shows how Chunk Key Phrases in medical texts can be extracted automatically using Spark NLP models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CHUNK_KEYWORD_EXTRACTOR/ 
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/9.Chunk_Key_Phrase_Extraction.ipynb
        - title: Recognize Clinical Abbreviations and Acronyms
          id: recognize_clinical_abbreviations_and_acronyms
          image: 
              src: /assets/images/Recognize_clinical_abbreviations_and_acronyms.svg
          excerpt: This demo shows how to extract clinical abbreviations and acronyms from medical texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_ABBREVIATION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_ABBREVIATION.ipynb
        - title: Link entities to Wikipedia pages
          id: link_entities_to_wikipedia_pages
          image: 
              src: /assets/images/Link_entities_to_Wikipedia_pages.svg
          excerpt: Automatically disambiguate people’s names based on their context and link them to corresponding Wikipedia pages using out of the box Spark NLP pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DISAMBIGUATION/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb        
---