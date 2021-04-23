---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /split_clean_medical_text
key: demo
license: false
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
          image2: 
              src: /assets/images/Detect_clinical_events_f.svg
          excerpt: Automatically identify from clinical documents using our pretrained Spark NLP model <b>ner_bionlp.</b>
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/CONTEXTUAL_SPELL_CHECKER
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CONTEXTUAL_SPELL_CHECKER.ipynb
        - title: Detect sentences in healthcare documents
          id: detect_sentences_in_healthcare_documents
          image: 
              src: /assets/images/Detect_sentences_in_healthcare_documents.svg
          image2: 
              src: /assets/images/Detect_sentences_in_healthcare_documents_f.svg
          excerpt: Automatically detect sentences in noisy healthcare documents with our pretrained Sentence Splitter DL model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/SENTENCE_DETECTOR_HC/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb
        - title: Normalize Medication-related Phrases
          id: normalize_medication-related_phrases
          image: 
              src: /assets/images/Normalize_Medication-related_Phrases.svg
          image2: 
              src: /assets/images/Normalize_Medication-related_Phrases_f.svg
          excerpt: Normalize medication-related phrases such as dosage, form and strength, as well as abbreviations in text and named entities extracted by NER models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/DRUG_NORMALIZATION
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/23.Drug_Normalizer.ipynb
        - title: Link entities to Wikipedia pages
          id: link_entities_to_wikipedia_pages
          image: 
              src: /assets/images/Link_entities_to_Wikipedia_pages.svg
          image2: 
              src: /assets/images/Link_entities_to_Wikipedia_pages_f.svg
          excerpt: Automatically disambiguate people’s names based on their context and link them to corresponding Wikipedia pages using out of the box Spark NLP pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DISAMBIGUATION/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb
---
