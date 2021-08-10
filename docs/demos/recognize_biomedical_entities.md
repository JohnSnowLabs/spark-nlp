---
layout: demopage
title: Spark NLP in Action
full_width: true
permalink: /recognize_biomedical_entities
key: demo
license: false
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Spark NLP for Healthcare 
      excerpt: Recognize Biomedical Entities 
      secheader: yes
      secheader:
        - title: Spark NLP for Healthcare
          subtitle: Recognize Biomedical Entities 
          activemenu: recognize_biomedical_entities
      source: yes
      source: 
        - title: Detect chemical compounds and genes
          id: detect_chemical_compounds_and_genes
          image: 
              src: /assets/images/Detect_chemical_compounds_and_genes.svg
          image2: 
              src: /assets/images/Detect_chemical_compounds_and_genes_f.svg
          excerpt: Automatically detect all chemical compounds and gene mentions using our pretrained chemprot model included in Spark NLP for Healthcare.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMPROT_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMPROT_CLINICAL.ipynb
        - title: Detect genes and human phenotypes
          id: detect_genes_and_human_phenotypes
          image: 
              src: /assets/images/Detect_genes_and_human_phenotypes.svg
          image2: 
              src: /assets/images/Detect_genes_and_human_phenotypes_f.svg
          excerpt: Automatically detect mentions of genes and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL.ipynb
        - title: Detect normalized genes and human phenotypes
          id: detect_normalized_genes_and_human_phenotypes
          image: 
              src: /assets/images/Detect_normalized_genes_and_human_phenotypes.svg
          image2: 
              src: /assets/images/Detect_normalized_genes_and_human_phenotypes_f.svg
          excerpt: Automatically detect normalized mentions of genes (go) and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL.ipynb
        - title: Detect cell structure, DNA, RNA and protein
          id: detect_cell_structure
          image: 
              src: /assets/images/Detect_cell_structure_DNA_RNA_and_protein.svg
          image2: 
              src: /assets/images/Detect_cell_structure_DNA_RNA_and_protein_f.svg
          excerpt: Automatically detect cell type, cell line, DNA and RNA information using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CELLULAR/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb
        - title: Detect chemical compounds
          id: detect_chemical_compounds
          image: 
              src: /assets/images/Detect_chemical_compounds.svg
          image2: 
              src: /assets/images/Detect_chemical_compounds_f.svg
          excerpt: Automatically detect all types of chemical compounds using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMICALS/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb
        - title: Detect bacteria, plants, animals or general species
          id: detect_bacteria
          image: 
              src: /assets/images/Detect_bacteria_plants_animals_or_general_species.svg
          image2: 
              src: /assets/images/Detect_bacteria_plants_animals_or_general_species_f.svg
          excerpt: Automatically detect bacteria, plants, animals, and other species using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_BACTERIAL_SPECIES/
          - text: Colab Netbook
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLINICAL_NER.ipynb 
---
