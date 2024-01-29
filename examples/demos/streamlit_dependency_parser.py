import svgwrite
import numpy as np
import pandas as pd
from streamlit_ner_output import get_color


def size(text):
  return ((len(text)+1)*15)-5

def draw_line(dwg, s_x , s_y, e_x, e_y, d_type, color):
  line = dwg.add(dwg.polyline(
      [(s_x, s_y+4), 
      (s_x, e_y), 
      (e_x, e_y), 
      (e_x, s_y),
      (e_x+2, s_y),
      (e_x, s_y+4),
      (e_x-2, s_y),
      (e_x, s_y)
      ],
      stroke=color, stroke_width = "2", fill='none',))
  dwg.add(dwg.text(d_type, insert=(((s_x+e_x)/2)-(size(d_type.strip())/2.75), e_y-4), 
  fill=color, font_size='20', font_family='courier'))

def generate_graph(result_df):
  # returns an svg graph

  colors_dict = {}
  max_x = 50
  max_y = 100
  
  for i in result_df['dependency_type'].unique():
    colors_dict[i] = get_color(i)
  
  for i in result_df['pos'].unique():
    colors_dict[i] = get_color(i)

  for i, row in result_df.iterrows():
    txt = row['chunk'].strip()
    max_x += (size(txt) + 50)
    max_y += 30
  
  max_x += 50
  start_x = 50
  starty_y = max_y
  dp_dict={}
  tk_dict = {}
  dist_dict = {}
  main_text = []
  main_pos = []

  for i, row in result_df.iterrows():
    txt = row['chunk'].strip()
    dt = row['dependency_type'].lower().strip()
    is_root = False
    if dt == 'root':
      is_root = True
    main_text.append((txt, start_x, starty_y, is_root))
    main_pos.append((row['pos'].strip(), (start_x + int((size(txt)/2) - int(size(row['pos'])/2))), starty_y+30))
    
    tk_dict[str(row['begin'])+str(row['end'])] = (start_x+int(size(txt)/2), starty_y)
    start_x += (size(txt) + 50)
  
  y_offset = starty_y-100
  dist_dict = {}
  e_dist_dict = {}
  direct_dict = {}
  left_side_dict = {}
  right_side_dict = {}
  y_hist = {}
  root_list = []
  main_lines = []
  lines_dist = []

  dist = []
  for i, row in result_df.iterrows():
    if row['dependency_type'].lower().strip() != 'root':
      lines_dist.append(abs(int(row['begin']) - int(row['dependency_start']['head.begin'])))
    else:
      lines_dist.append(0)
      
  result_df = result_df.iloc[np.argsort(lines_dist)]

  count_left = {}
  count_right = {}
  t_x_offset = {}
  for i, row in result_df.iterrows():
    if row['dependency_type'].lower().strip() != 'root':
      sp = str(row['dependency_start']['head.begin'])+str(row['dependency_start']['head.end'])
      x_e, y_e = tk_dict[str(row['begin'])+str(row['end'])]
      x, y = tk_dict[sp]
      if int(row['begin']) < int(row['dependency_start']['head.begin']):
        if x in count_left:
          count_left[x] += 1
          t_x_offset[x] += 7
        else:
          count_left[x] = 1
          t_x_offset[x] = 7
        if x_e in count_right:
          count_right[x_e] += 1
          t_x_offset[x_e] -= 7
        else:
          count_right[x_e] = 0
          t_x_offset[x_e] = 0
      else:
        if x in count_right:
          count_right[x] += 1
          t_x_offset[x] -= 7
        else:
          count_right[x] = 0
          t_x_offset[x] = 0
        if x_e in count_left:
          count_left[x_e] += 1
          t_x_offset[x_e] += 7
        else:
          count_left[x_e] = 1
          t_x_offset[x_e] = 7
  
  for i, row in result_df.iterrows():
    
    sp = str(row['dependency_start']['head.begin'])+str(row['dependency_start']['head.end'])
    ep = tk_dict[str(row['begin'])+str(row['end'])]

    if sp != '-1-1':
      x, y = tk_dict[sp]

      if int(row['begin']) > int(row['dependency_start']['head.begin']):
        dist_dict[x] = count_right[x] * 7
        count_right[x] -= 1
        e_dist_dict[ep[0]] = count_left[ep[0]] * -7
        count_left[ep[0]] -= 1
      else:
        dist_dict[x] = count_left[x] * -7
        count_left[x] -= 1
        e_dist_dict[ep[0]] = count_right[ep[0]] * 7
        count_right[ep[0]] -= 1
      #row['dependency'], x, t_x_offset[x], x+dist_dict[x], x+dist_dict[x]+t_x_offset[x]
      final_x_s = int(x+dist_dict[x]+(t_x_offset[x]/2))
      final_x_e = int(ep[0]+ e_dist_dict[ep[0]]+(t_x_offset[ep[0]]/2))

      x_inds = range(min(final_x_s, final_x_e), max(final_x_s, final_x_e)+1)
      common = set(y_hist.keys()).intersection(set(x_inds))

      if common:
        y_fset = min([y_hist[c] for c in common])
        y_fset -= 50
        y_hist.update(dict(zip(x_inds, [y_fset]*len(x_inds))))
        
      else:
        y_hist.update(dict(zip(x_inds, [y_offset]*len(x_inds))))

      main_lines.append((None, final_x_s, y-30, final_x_e, y_hist[final_x_s], row['dependency_type']))

    else:
      x_x , y_y = tk_dict[str(row['begin'])+str(row['end'])]

      root_list.append((row['dependency_type'].upper(), x_x, y_y))


  current_y = min(y_hist.values())

  y_ff = (max_y - current_y) + 50
  y_f = (current_y - 50)
  current_y = 50

  dwg = svgwrite.Drawing("temp.svg",
                        profile='tiny', size = (max_x, y_ff+100))

  for mt, mp in zip(main_text, main_pos):
    dwg.add(dwg.text(mt[0], insert=(mt[1], mt[2]-y_f), fill='gray', 
    font_size='25', font_family='courier'))

    if mt[3]:
      dwg.add(dwg.rect(insert=(mt[1]-5, mt[2]-y_f-25), size=(size(mt[0]),35), stroke='orange', 
      stroke_width='2', fill='none'))

    dwg.add(dwg.text(mp[0], insert=(mp[1], mp[2]-y_f), fill=colors_dict[mp[0]]))

  for ml in main_lines:
    draw_line(dwg, ml[1], ml[2]-y_f, ml[3], ml[4]-y_f, ml[5], colors_dict[ml[5]])
  
  return dwg.tostring()
