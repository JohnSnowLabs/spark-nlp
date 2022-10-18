---
layout: demopagenew
title: Spark NLP in Action
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /split_clean_medical_text
key: demo
article_header:
  type: demo
license: false
mode: immersivebg
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Split & Clean Medical Text 
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Split & Clean Medical Text 
          activemenu: split_clean_medical_text
      source: yes
      source: 
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
---
