import streamlit as st
import random
import base64
import pandas as pd
import numpy as np
import streamlit_apps_config as config
# from colour import Color
current_path = config.project_path
def get_color(l):
    if str(l).lower() in config.LABEL_COLORS.keys():
        return config.LABEL_COLORS[l.lower()]
    else:
        r = lambda: random.randint(0,200)
        return '#%02X%02X%02X' % (r(), r(), r())


def jsl_display_annotations_not_converted(original_text, fully_annotated_text, labels):
    """Function to display NER annotation when ner_converter was not used
    """
    label_color = {}
    for l in labels:
        label_color[l] = get_color(l)
    html_output = ""
    #html_output = """<div>"""
    pos = 0
    for n in fully_annotated_text['ner']:
        begin = n[1]
        end = n[2]
        entity = n[3] # When ner_converter: n[4]['entity']
        word = n[4]['word'] # When ner_converter: n[3]
        if pos < begin and pos < len(original_text):
            white_text = original_text[pos:begin]
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(white_text)
        pos = end+1

        if entity in label_color:
            html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span><span class="entity-type">{}</span></span>'.format(
                label_color[n[3]],
                word,
                entity)
        else:
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(word)

    if pos < len(original_text):
        html_output += '<span class="others" style="background-color: white">{}</span>'.format(original_text[pos:])

    html_output += """</div>"""
    return html_output


def jsl_display_annotations(original_text, fully_annotated_text, labels):
    label_color = {}
    for l in labels:
        label_color[l] = get_color(l)
    html_output = ""
    #html_output = """<div>"""
    pos = 0
    for n in fully_annotated_text['ner_chunk']:
        #print (n)
        begin = n[1]
        end = n[2]
        if pos < begin and pos < len(original_text):
            white_text = original_text[pos:begin]
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(white_text)
        pos = end+1

        if n[4]['entity'] in label_color:
            html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span><span class="entity-type">{}</span></span>'.format(
                label_color[n[4]['entity']],
                n[3],
                n[4]['entity'])
        else:
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(n[3])

    if pos < len(original_text):
        html_output += '<span class="others" style="background-color: white">{}</span>'.format(original_text[pos:])

    html_output += """</div>"""
    return html_output


def show_html2(original_text, fully_annotated_text, label_set, title_message="Text annotated with identified Named Entities", show_tag=True, converted=True):
    """Show annotation as HTML objects

    David Cecchini: Added the parameter `converted` to control if the annotated text is output of ner_converter or not (use nerTagger output)
    """

    if show_tag is False:
        st.subheader("Text annotated with matched Entities".format(''))
        html_content = jsl_display_annotations_without_tag(original_text, fully_annotated_text, label_set)
        html_content = html_content.replace("\n", "<br>")
        st.write(config.HTML_WRAPPER.format(html_content), unsafe_allow_html=True)
    else:
        #st.subheader("Text annotated with identified Named Entities".format(''))
        st.subheader(title_message.format(''))
        if converted:
            html_content = jsl_display_annotations(original_text, fully_annotated_text, label_set)
        else:
            html_content = jsl_display_annotations_not_converted(original_text, fully_annotated_text, label_set)
        html_content = html_content.replace("\n", "<br>")
        st.write(config.HTML_WRAPPER.format(html_content), unsafe_allow_html=True)

    st.write('')

def jsl_display_annotations_without_tag(original_text, fully_annotated_text, labels):
    label_color = {}
    for l in labels:
        label_color[l] = get_color(l)
    html_output = ""
    #html_output = """<div>"""
    pos = 0
    for n in fully_annotated_text['matched_text']:
        #print (n)
        begin = n[1]
        end = n[2]
        if pos < begin and pos < len(original_text):
            white_text = original_text[pos:begin]
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(white_text)
        pos = end+1

        if n[3] in label_color:
            html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span></span>'.format(
                label_color[n[3]],
                n[3])
        else:
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(n[3])

    if pos < len(original_text):
        html_output += '<span class="others" style="background-color: white">{}</span>'.format(original_text[pos:])

    html_output += """</div>"""
    return html_output

def jsl_display_spell_correction(original_tokens, corrected_tokens):
  
  color = get_color('rand')

  st.subheader("Text annotated with corrected words".format(''))

  html_output = ''
  for original_token, corrected_token in zip(original_tokens, corrected_tokens):
    original = original_token[3]
    corrected = corrected_token[3]
    if original != corrected:   
      html_output += ' <span class="entity-wrapper" style="background-color: {}"><span class="entity-name"><del> {} </del> {} </span></span>'.format(color, original, corrected)
    
    
    else:
      original = original if original in set([",", "."]) else ' ' + original #quick and dirty handle formatting
      html_output += '<span class="others" style="background-color: white">{}</span>'.format(original)


  html_output = html_output.replace("\n", "<br>")
  st.write(config.HTML_WRAPPER.format(html_output), unsafe_allow_html=True)
    
    
def jsl_display_entity_resolution(original_text, fully_annotated_text, labels):
    label_color = {}
    for l in labels:
        label_color[l] = get_color(l)
    html_output = ""
    #html_output = """<div>"""
    pos = 0
    for i, n in fully_annotated_text.iterrows():
        begin = n[1]
        end = n[2]
        if pos < begin and pos < len(original_text):
            white_text = original_text[pos:begin]
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(white_text)
        pos = end+1
        
        resolution_chunk = n[4]
        resolution_exp = n[5]
        if n[3] in label_color:
            second_color = get_color(resolution_chunk)
            if resolution_exp.lower() != 'na':
                html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span><span class="entity-type">{}</span><span class="entity-type" style="background-color: {}">{} </span><span class="entity-type" style="background-color: {}">{}</span></span>'.format(
                    label_color[n[3]] + 'B3', #color
                    n[0], #entity - chunk
                    n[3], #entity - label
                    label_color[n[3]] + 'FF', #color '#D2C8C6' 
                    resolution_chunk, # res_code
                    label_color[n[3]] + 'CC', # res_color '#DDD2D0'
                    resolution_exp) # res_text
                
            else:
                html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span><span class="entity-type">{}</span></span>'.format(
                        label_color[n[3]],
                        n[0],
                        n[3])
        
    if pos < len(original_text):
        html_output += '<span class="others" style="background-color: white">{}</span>'.format(original_text[pos:])

    html_output += """</div>"""
    html_output = html_output.replace("\n", "<br>")
    st.write(config.HTML_WRAPPER.format(html_output), unsafe_allow_html=True)
    
def jsl_display_assertion(original_text, fully_annotated_text, labels):
    label_color = {}
    for l in labels:
        label_color[l] = get_color(l)
    html_output = ""
    #html_output = """<div>"""
    pos = 0
    for i, n in fully_annotated_text.iterrows():
        begin = n[1]
        end = n[2]
        if pos < begin and pos < len(original_text):
            white_text = original_text[pos:begin]
            html_output += '<span class="others" style="background-color: white">{}</span>'.format(white_text)
        pos = end+1
        
        resolution_chunk = n[4]
        if n[3] in label_color:
            if resolution_chunk.lower() != 'na':
                html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span><span class="entity-type">{}</span><span class="entity-type" style="background-color: {}">{} </span></span>'.format(
                    label_color[n[3]] + 'B3', #color
                    n[0], #entity - chunk
                    n[3], #entity - label
                    label_color[n[3]] + 'FF', #color '#D2C8C6' 
                    resolution_chunk)
            else:
                html_output += '<span class="entity-wrapper" style="background-color: {}"><span class="entity-name">{} </span><span class="entity-type">{}</span></span>'.format(
                        label_color[n[3]],
                        n[0],
                        n[3])
        
    if pos < len(original_text):
        html_output += '<span class="others" style="background-color: white">{}</span>'.format(original_text[pos:])

    html_output += """</div>"""
    html_output = html_output.replace("\n", "<br>")
    st.write(config.HTML_WRAPPER.format(html_output), unsafe_allow_html=True)
    
def display_example_text(text):
    return """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem; white-space:pre-wrap; min-height: 200px; max-height: 500px; line-height: 2.0">{}</div>""".format(text)
