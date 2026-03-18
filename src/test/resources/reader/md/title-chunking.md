# Overview

Unstructured can parse Markdown into elements.
This makes it easy to experiment with chunking locally.

## Configuration

max_characters controls hard chunk size.

new_after_n_chars controls a softer preferred limit.

combine_text_under_n_chars helps avoid tiny chunks.

## Example

When the parser detects headings, they become Title elements.
The chunk_by_title function uses those Title elements as section boundaries.
