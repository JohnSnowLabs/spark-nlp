# How to build the PyDocs

0. Change directories to this folder `spark-nlp/python/docs`
1. Install requirements `requirements_doc.txt`
2. run `make html`
   1. This will generate the html and move it to `../../docs/api/python`

The html will be available under `../../docs/api/python/index.html`.

## Note
The folder `_autosummary` should not be committed, as it is generated from
sphinx itself.