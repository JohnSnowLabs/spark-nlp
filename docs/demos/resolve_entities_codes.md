---
layout: demopagenew
title: Resolve Entities to Codes - Clinical NLP Demos & Notebooks
seotitle: 'Clinical NLP: Resolve Entities to Codes - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /resolve_entities_codes
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
        - subtitle: Resolve Entities to Codes - Live Demos & Notebooks
          activemenu: resolve_entities_codes
      source: yes
      source: 
        - title: Map clinical terminology to SNOMED taxonomy
          id: snomed_coding
          image: 
              src: /assets/images/Detect_signs_and_symptoms.svg
          excerpt: Automatically resolve the SNOMED code corresponding to the diseases and conditions mentioned in your health record using Spark NLP for Healthcare out of the box.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_SNOMED
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_SNOMED.ipynb        
        - title: Map clinical terminology to ICD-10-CM taxonomy
          id: icd10-cm_coding
          image: 
              src: /assets/images/Resolve_Symptoms_to_ICD10-CM_Codes.svg
          excerpt: This demo shows how clinical problems can be automatically mapped to the ICD10-CM taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb
        - title: Map drug terminology to RxNorm taxonomy
          id: rxnorm_coding
          image: 
              src: /assets/images/Resolve_Drugs_to_RxNorm_Codes.svg
          excerpt: This demo shows how drugs can be automatically mapped to RxNorm codes using sentence based resolvers. 
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_RXNORM/
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_RXNORM.ipynb              
        - title: Map healthcare codes between taxonomies
          id: logical-observation-identifiers-names-and-codes
          image: 
              src: /assets/images/Map_Healthcare_Codes.svg
          excerpt: These pretrained pipelines map various codes (e.g., ICD10CM codes to SNOMED codes) without using any text data.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_CODE_MAPPING/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.1.Healthcare_Code_Mapping.ipynb#scrollTo=e5qYdIEv4JPL
        - title: Map laboratory terminology to LOINC taxonomy
          id: resolve_clinical_entities_codes_loinc
          image: 
              src: /assets/images/Resolve_Clinical_Entities_to_LOINC.svg
          excerpt: This demo shows how laboratory terminology can be automatically mapped to the Logical Observation Identifiers Names and Codes (LOINC) taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_LOINC/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_LOINC.ipynb        
        - title: Resolve Clinical Health Information using the HPO taxonomy
          id: resolve_clinical_health_information_using_hpo_taxonomy 
          image: 
              src: /assets/images/Resolve_Clinical_Health_Information_using_the_HPO_taxonomy.svg
          excerpt: This demo shows how clinical health information can be mapped to codes using the HPO taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_HPO/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_HPO.ipynb
        - title: Resolve Clinical Health Information using the MeSH taxonomy
          id: resolve_clinical_health_information_using_mesh_taxonomy 
          image: 
              src: /assets/images/Resolve_Clinical_Health_Information_using_the_MeSH_taxonomy.svg
          excerpt: This demo shows how clinical health information can be mapped to codes using the MeSH taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_MSH/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_MSH.ipynb
        - title: Resolve Clinical Findings using the UMLS CUI taxonomy
          id: resolve_clinical_health_information_using_umls_cui_taxonomy 
          image: 
              src: /assets/images/Sentence_Entity_Resolver_for_UMLS_CUI.svg
          excerpt: This demo shows how clinical findings can be mapped to codes using the UMLS CUI taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_UMLS_CUI/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_UMLS_CUI.ipynb
        - title: Map clinical terminology to HCPCS taxonomy
          hide: yes
          id: map_clinical_terminology_hcpcs_taxonomy  
          image: 
              src: /assets/images/Map_clinical_terminology_to_HCPCS_taxonomy.svg
          excerpt: This demo shows how clinical terminology can be automatically mapped to the Healthcare Common procedure Coding System (HCPCS) taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_HCPCS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_HCPCS.ipynb
        - title: Resolve Clinical Health Information using the NDC taxonomy
          id: resolve_clinical_health_information_using_ndc_taxonomy   
          image: 
              src: /assets/images/Resolve_Clinical_Health_Information_using_the_NDC_taxonomy.svg
          excerpt: This demo shows how clinical health information and concepts (like drugs/ingredients) can be mapped to codes using the NDC taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_NDC/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_NDC.ipynb#scrollTo=dnJ9X-mbEOMr
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
        - title: Resolve Drug Class using RxNorm taxonomy
          id: resolve_drug_class_using_rxnorm_taxonomy     
          image: 
              src: /assets/images/Resolve_Drug_Class_using_RxNorm_taxonomy.svg
          excerpt: This demo shows how to map Drugs to related Drug-Classes using RxNorm taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_RXNORM_DRUG_CLASS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_RXNORM_DRUG_CLASS.ipynb
        - title: Resolve Drug & Substance using the UMLS CUI taxonomy
          id: resolve_drug_Substance_using_umls_cui_taxonomy      
          image: 
              src: /assets/images/Resolve_Drug_Substance_using_the_umls_cuitaxonomy.svg
          excerpt: This demo shows how to map Drug & Substance to their corresponding codes using UMLS CUI taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_UMLS_CUI_DRUG_SUBSTANCE/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_UMLS_CUI_DRUG_SUBSTANCE.ipynb
        - title: Resolve Clinical Procedures using CPT taxonomy
          id: resolve_clinical_procedures_cpt_taxonomy       
          image: 
              src: /assets/images/Resolve_Clinical_Procedures_using_CPT_taxonomy.svg
          excerpt: This demo shows how to map clinical procedures to codes using CPT taxonomy.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/ER_CPT/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_CPT.ipynb
---
