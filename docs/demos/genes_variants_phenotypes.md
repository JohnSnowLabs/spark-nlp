---
layout: demopagenew
title: Genes, Variants, Phenotypes - Biomedical NLP Demos & Notebooks
seotitle: 'Biomedical NLP: Genes, Variants, Phenotypes - John Snow Labs'
subtitle: Run 300+ live demos and notebooks
full_width: true
permalink: /genes_variants_phenotypes
key: demo
article_header:
  type: demo
license: false
mode: immersivebg
show_edit_on_github: false
show_date: false
data:
  sections:  
    - title: Biomedical NLP
      excerpt: Genes, Variants, Phenotypes
      secheader: yes
      secheader:
        - title: Biomedical NLP
          subtitle: Genes, Variants, Phenotypes
          activemenu: genes_variants_phenotypes
      source: yes
      source:
        - title: Detect chemical compounds and genes
          id: detect_chemical_compounds_and_genes
          image: 
              src: /assets/images/Detect_chemical_compounds_and_genes.svg
          excerpt: Automatically detect all chemical compounds and gene mentions using our pretrained chemprot model included in Spark NLP for Healthcare.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMPROT_CLINICAL
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMPROT_CLINICAL.ipynb
        - title: Detect genes and human phenotypes
          id: detect_genes_and_human_phenotypes
          image: 
              src: /assets/images/Detect_genes_and_human_phenotypes.svg
          excerpt: Automatically detect mentions of genes and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL.ipynb
        - title: Detect normalized genes and human phenotypes
          id: detect_normalized_genes_and_human_phenotypes
          image: 
              src: /assets/images/Detect_normalized_genes_and_human_phenotypes.svg
          excerpt: Automatically detect normalized mentions of genes (go) and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL.ipynb
        - title: Detect cell structure, DNA, RNA and protein
          id: detect_cell_structure
          image: 
              src: /assets/images/Detect_cell_structure_DNA_RNA_and_protein.svg
          excerpt: Automatically detect cell type, cell line, DNA and RNA information using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CELLULAR/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CELLULAR.ipynb        
        - title: Detect bacteria, plants, animals or general species
          id: detect_bacteria
          image: 
              src: /assets/images/Detect_bacteria_plants_animals_or_general_species.svg
          excerpt: Automatically detect bacteria, plants, animals, and other species using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_BACTERIAL_SPECIES/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BACTERIAL_SPECIES.ipynb
        - title: Detect Genomic Variant Information
          id: detect_genomic_variant_information 
          image: 
              src: /assets/images/Detect_Genomic_Variant_Information.svg
          excerpt: This model extracts genetic variant information from the medical text.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_GEN_VAR/
          - text: Colab
            type: blue_btn
            url: https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb
        - title: Extract chemical compounds, drugs, genes and proteins 
          id: extract_chemical_compounds_drugs_genes_proteins  
          image: 
              src: /assets/images/Extract_chemical_compounds_drugs_genes_and_proteins.svg
          excerpt: This demo shows how to extract chemical compounds, drugs, genes and proteins from medical texts.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_DRUG_PROT/
          - text: Colab
            type: blue_btn
            url: https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb
        - title: Detect chemical compounds
          id: detect_chemical_compounds
          image: 
              src: /assets/images/Detect_chemical_compounds.svg
          excerpt: Automatically detect all types of chemical compounds using our pretrained Spark NLP for Healthcare model.
          actions:
          - text: Live Demo
            type: normal
            url: https://demo.johnsnowlabs.com/healthcare/NER_CHEMICALS/
          - text: Colab
            type: blue_btn
            url: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMICALS.ipynb      
---
