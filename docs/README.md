# Spark NLP Documentation

We welcome you to contribute to Spark NLP documentation hosted inside `en/` directory. All the files are in Markdown format.

## Development

For development purposes, you need to have `bundle` and `Gem` installed on your system. Please run these commands:

```bash
bundle update
bundle install
bundle exec jekyll serve

# Server address: http://127.0.0.1:4000
```

## How to build the PyDocs

1. Install requirements `requirements_doc.txt`
2. run `make html`

The html will be available under `_build/html/index.html`.

## Note

The folder `_autosummary` should not be committed, as it is generated from sphinx itself.
