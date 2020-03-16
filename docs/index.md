---
layout: landing
title: 'Spark NLP: State of the Art <br /> Natural Language Processing'
excerpt: >
   <br> The first production grade versions of the latest deep learning NLP research
permalink: /
header: true
article_header:
  actions:
    - text: Getting Started
      type: error
      url: /docs/en/quickstart    
    - text: '<i class="fab fa-github"></i> GitHub'
      type: outline-theme-dark
      url: https://github.com/johnsnowlabs/spark-nlp  
    - text: '<i class="fab fa-slack-hash"></i> Slack' 
      type: outline-theme-dark
      url: https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLWVjNWUzOGNlODg1Y2FkNGEzNDQ1NDJjMjc3Y2FkOGFmN2Q3ODIyZGVhMzU0NGM3NzRjNDkyZjZlZTQ0YzY1N2I    
  height: 50vh
  theme: dark
  background_color: "#0296D8"
  # background_image:
    # gradient: "linear-gradient(rgba(0, 0, 0, .2), rgba(0, 0, 0, .6))"
data:
  sections:
    - title: <h3>The most widely used NLP library in the enterprise</h3>
      excerpt: Backed by <b>O'Reilly's</b> most recent "AI Adoption in the Enterprise" survey in February
      children:
        - title: 100% Open Source
          excerpt: Including pre-trained <b>models</b> and <b>pipelines</b>
        - title: Natively scalable
          excerpt: The only <b>NLP</b> library built <b>natively</b> on Apache Spark   
        - title: Multiple Languages
          excerpt: Full <b>Python</b>, <b>Scala</b>, and <b>Java</b> support
   
    - title: '<h2> Quick and Easy </h2>'
      install: yes
      excerpt: Spark NLP is available on <a href="https://pypi.org/project/spark-nlp" target="_blank">PyPI</a>, <a href="https://anaconda.org/JohnSnowLabs/spark-nlp" target="_blank">Conda</a>, <a href="https://mvnrepository.com/artifact/JohnSnowLabs/spark-nlp" target="_blank">Maven</a>, and <a href="https://spark-packages.org/package/JohnSnowLabs/spark-nlp" target="_blank">Spark Packages</a>
      background_color: "#ecf0f1"
      actions:
        - text: Install Spark NLP
          url: /docs/en/install
    

    - title: Right Out of The Box
      excerpt: Spark NLP ships with many <b>NLP features</b>, pre-trained <b>models</b> and <b>pipelines</b>
      actions:
        - text: Pipelines
          url: /docs/en/pipelines     
        - text: Models
          url: /docs/en/models
      features: true
      # theme: dark
      # background_color: "#123"
    
    - title: Benchmark
      excerpt: Spark NLP 2.4.x obtained the best performing academic peer-reviewed results
      benchmark: yes
      features: false
      theme: dark
      background_color: "#123"

    - title: TRUSTED BY
      background_color: "#ffffff"
      children:
        - title:
          image: 
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_logo_%282012%29.svg/500px-Microsoft_logo_%282012%29.svg.png
            url: https://www.microsoft.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/500px-Google_2015_logo.svg.png
            url: https://cloud.google.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/500px-Amazon_Web_Services_Logo.svg.png
            url: https://aws.amazon.com/
            style: "max-width: 120px; max-height: 120px"
            is_row: true        
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Intel-logo.svg/500px-Intel-logo.svg.png
            url: https://www.intel.com/
            style: "max-width: 150px; max-height: 150px"
            is_row: true
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/500px-IBM_logo.svg.png
            url: https://www.ibm.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true   
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/f/fa/Indeed_logo.png
            url: https://www.indeed.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true  
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Roche_Logo.svg/500px-Roche_Logo.svg.png
            url: https://www.roche.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/3/34/DocuSign_logo.png
            url: https://www.docusign.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Viacom_logo.svg/500px-Viacom_logo.svg.png
            url: https://www.viacbs.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Capital_One_logo.svg/500px-Capital_One_logo.svg.png
            url: https://www.capitalone.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true            
        - title:
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Verizon_2015_logo_-vector.svg/500px-Verizon_2015_logo_-vector.svg.png
            url: https://www.verizonwireless.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title: 
          image:
            src: https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Mck_logo_pos_col_rgb.svg/500px-Mck_logo_pos_col_rgb.svg.png
            url: https://www.mckesson.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true    
        - title: 
          image:
            src: https://selectdata.com/wp-content/uploads/2019/12/logo.png
            url: https://selectdata.com/
            style: "max-width: 200px; max-height: 200px"
            is_row: true
        - title: 
          image:
            src: http://www.cnrs.fr/themes/custom/cnrs/logo.svg
            url: https://iscpif.fr/
            style: "max-width: 110px; max-height: 110px;margin-bottom:10px"
            is_row: true        
          
    - title: <h2>Active Community Support</h2>
      theme: dark
      excerpt: 
      actions:        
        - text: '<i class="fas fa-terminal"></i> Examples'
          type: outline-theme-dark
          url: https://github.com/JohnSnowLabs/spark-nlp-workshop
        - text: '<i class="fab fa-slack-hash"></i> Slack'
          type: outline-theme-dark
          url: https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLWVjNWUzOGNlODg1Y2FkNGEzNDQ1NDJjMjc3Y2FkOGFmN2Q3ODIyZGVhMzU0NGM3NzRjNDkyZjZlZTQ0YzY1N2I
        - text: '<iframe src="https://ghbtns.com/github-btn.html?user=johnsnowlabs&repo=spark-nlp&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>'
          type: dark
          url: https://github.com/johnsnowlabs/spark-nlp    
      background_color: "#0296D8"
    
    
---