import os
PRODUCTION = False
if PRODUCTION:
    prepath = "/Users/vkocaman/cache_pretrained/"
    path = "/Users/vkocaman/cache_pretrained/2.4/"
    jar_path = "/home/ubuntu/jars/"
    input_folder = '/home/ubuntu/streamlit/dia/resources/'
    rules_folder = '/home/ubuntu/streamlit/dia/resources/rules/'
else:
    project_path = "F:/JSL/streamlit-demo-apps/"

    prepath = "file:///E://JSL/SparkNLPSUITE/models/cache_pretrained/"
    path = "file:///E://JSL/SparkNLPSUITE/models/cache_pretrained/2.4/"
    jar_path = "file:///E:/JSL/Jars/streamlit4/"
    rules_folder = "file:///E://JSL/SparkNLPSUITE/streamlit-demo-apps/resources/rules/"
    models_folder = "file:///E://JSL/SparkNLPSUITE/models/"

ENTITIES_FOR_ICD10 = ['problem', 'diagnosis', 'procedure name', 'lab name', 'symptom_name', 'procedure_name', 'procedure', 'lab_name', 'pathological_formation', 'cancer']
LOGO_PATH = '../resources/jsl-logo.png'
available_models = []

#APP STYLE
MAX_WIDTH = 1600
PADDING_TOP = 1
PADDING_BOTTOM = 1
PADDING_RIGHT = 4
PADDING_LEFT = 4
COLOR = 'black'
BACKGROUND_COLOR = 'white'

HTML_WRAPPER = """<div class="scroll entities" style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem; white-space:pre-wrap">{}</div>"""
HTML_INDEX_WRAPPER = """<div ">{}</div>"""

STYLE_CONFIG_OLD = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    *:not(text){{
      font-family: Montserrat;
    }}
    
    .reportview-container .main .block-container{{
        max-width: {MAX_WIDTH}px;
        padding-top: {PADDING_TOP}rem;
        padding-right: {PADDING_RIGHT}rem;
        padding-left: {PADDING_LEFT}rem;
        padding-bottom: {PADDING_BOTTOM}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
    div.scroll {{ 
                margin:4px, 4px; 
                padding:4px; 
                width: 100%; 
                height: 500px; 
                overflow-x: hidden; 
                overflow-x: auto;  
    }}
    .entity-wrapper{{
        padding: 5px;
        display: inline-grid;
        text-align:center;
        margin-bottom:5px;
        border-radius: 5px 5px
    }}
    .entity-name{{
        background: #f1f2f3;
        color: #3c3e44;
        padding: 2px;
        border-color: #484b51;
        border-width: medium;
        border-radius: 5px 5px;
    }}
    .entity-type{{
        color: #272727;
        text-transform: uppercase;
        font-family: roboto;
        font-size: 13px;
    }}
    .reportview-container .markdown-text-container{{
        font-family: roboto !important;
        color: dimgray !important;
        line-height: normal !important;
    }}s
    .reportview-container h2
    {{
        font-weight: 400 !important;
        font-size: 1.5rem !important;
        line-height: 1.6!important;
    }}
    .reportview-container h2
    {{
        font-weight: 300 !important;
        font-size: 1.3rem !important;
        line-height: 1.4!important;
    }}
    
    
</style>
"""

with open(rf'C:\Users\bdllh\Desktop\Spark NLP new features\TEXT_SUMMARIZATION_WITH_BART\style.css') as f:
    STYLE_CONFIG_NEW = f.read()
STYLE_CONFIG = STYLE_CONFIG_OLD + '<style>{}</style>'.format(STYLE_CONFIG_NEW)

LABEL_COLORS = {'problem':'#0C8888',
               'test':'#FF33C1',
               'treatment':'#3196D4',
                'multi':'#ccfff5',
                'multi-tissue_structure':'#8dd8b4',
                'cell':'#ffe6cc',
                'organism':'#ffddcc',
                'gene_or_gene_product':'#fff0b3',
                'organ':'#e6e600',
                'simple_chemical':'#ffd699',
                
                'per':'#0C8888', 'pers':'#0C8888','person':'#0C8888',
                'org':'#FF33C1',
                'misc': '#3196D4', 'mis': '#3196D4',
                'loc':'#5B00A3', 'location':'#5B00A3',
        
                'drug':'#33BBFF',
                'diagnosis':'#b5a1c9',
                'maybe':'#FFB5C5',
                'lab_result':'#3abd80',
                'negated':'#CD3700',
                'lab_name':'#698B22',
                'modifier':'#8B475D',
                'symptom_name':'#CDB7B5',
                'section_name':'#8B7D7B',
                'procedure_name':'#48D1CC',
                'grading':"#8c61e8",
                'size':"#746b87",
                'organism_substance':'#ffaa80',
                'gender':'#969c63',
                'age':'#6f6798',
                'date': '#4b7bb4',
                'name':'#95856a',
                
                'disease' : '#48b1b7',
                'test_result' : '#36454f',
                'symptom' : '#03402b',
                'drug_ingredient' : '#6f7590',
                'drug_brandname': '#6666b0',
                'relativetime':'#816798',
                'relativedate' : '#ab9754',
                'section_header':'#c4453b',
                'frequency': '#962424',
                'strength':'#4040b3',
                'route':'#2da894',
                'procedure':'#F9A851',
                'disease_syndrome_disorder':'#a87f0c',
                'oncological':'#656d78',
                'drug_ingredient':'#2d4f7d',
                'clinical_dept':'#aa5562',
                'patient':'#191970',
                'hospital':'#4a9ab5',
                'doctor' : '#522a59',
                'profession':'#b06e15',
                'ade' : '#048a5c',
                'dosage':'#9e7061',
                'form':'#8e6897',
                'abbreviation':'#759768',
                'condition':'#c64d39',
                'drugchem':'#5e6b5e',
                'measurement':'#6a8695',
                'bodypart':'#AF9674',
                'imagingfindings':'#53ac98',
                'cancer_surgery':'#a71fbf',
                'pathology_test' : '#5bae51',
                'chemotherapy':'#e0551f',
                'cancer_dx':'#cd327e',
                'biomarker_result':'#5c5656',
                'tumor_finding': '#402dd2',
                'anatomical_site':'#8f2b3a',
                'smoking_status' : '#bf5858',
                'cancer_therapy':'#8a9c63'

                }