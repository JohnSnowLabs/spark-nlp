"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as 'Required' below must be included for upload to PyPI.
# Fields marked as 'Optional' may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='spark-nlp',  # Required

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html

    version='4.2.1',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the 'Summary' metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='John Snow Labs Spark NLP is a natural language processing library built on top of Apache Spark ML. It provides simple, performant & accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.',  # Required

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the 'Description' metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, 'applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst' (see link below)
    #
    # This field corresponds to the 'Description-Content-Type' metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the 'Home-Page' metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/JohnSnowLabs/spark-nlp',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='John Snow Labs',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    # author_email='pypa-dev@googlegroups.com',  # Optional

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: English',
        'Natural Language :: Multilingual',
        'Natural Language :: Afrikaans',
        'Natural Language :: Afro-Asiatic languages',
        'Natural Language :: Albanian',
        'Natural Language :: Altaic languages',
        'Natural Language :: American Sign Language',
        'Natural Language :: Amharic',
        'Natural Language :: Arabic',
        'Natural Language :: Armenian',
        'Natural Language :: Artificial languages',
        'Natural Language :: Assamese',
        'Natural Language :: Atlantic-Congo languages',
        'Natural Language :: Austro-Asiatic languages',
        'Natural Language :: Austronesian languages',
        'Natural Language :: Azerbaijani',
        'Natural Language :: Baltic languages',
        'Natural Language :: Bantu languages',
        'Natural Language :: Basque',
        'Natural Language :: Basque (family)',
        'Natural Language :: Bemba (Zambia)',
        'Natural Language :: Bengali, Bangla',
        'Natural Language :: Berber languages',
        'Natural Language :: Bislama',
        'Natural Language :: Bosnian',
        'Natural Language :: Brazilian Sign Language',
        'Natural Language :: Bulgarian',
        'Natural Language :: Burmese',
        'Natural Language :: Catalan',
        'Natural Language :: Caucasian languages',
        'Natural Language :: Cebuano',
        'Natural Language :: Celtic languages',
        'Natural Language :: Central Bikol',
        'Natural Language :: Chichewa, Chewa, Nyanja',
        'Natural Language :: Chinese',
        'Natural Language :: Chuukese',
        'Natural Language :: Congo Swahili',
        'Natural Language :: Croatian',
        'Natural Language :: Cushitic languages',
        'Natural Language :: Czech',
        'Natural Language :: Danish',
        'Natural Language :: Dholuo, Luo (Kenya and Tanzania)',
        'Natural Language :: Dravidian languages',
        'Natural Language :: Dutch',
        'Natural Language :: East Slavic languages',
        'Natural Language :: Eastern Malayo-Polynesian languages',
        'Natural Language :: Efik',
        'Natural Language :: Esperanto',
        'Natural Language :: Estonian',
        'Natural Language :: Fijian',
        'Natural Language :: Finnish',
        'Natural Language :: Finno-Ugrian languages',
        'Natural Language :: French',
        'Natural Language :: French-based creoles and pidgins',
        'Natural Language :: Galician',
        'Natural Language :: Ganda',
        'Natural Language :: Georgian',
        'Natural Language :: German',
        'Natural Language :: Germanic languages',
        'Natural Language :: Gilbertese',
        'Natural Language :: Greek (modern)',
        'Natural Language :: Greek languages',
        'Natural Language :: Gujarati',
        'Natural Language :: Haitian, Haitian Creole',
        'Natural Language :: Hausa',
        'Natural Language :: Hebrew (modern)',
        'Natural Language :: Hiligaynon',
        'Natural Language :: Hindi',
        'Natural Language :: Hiri Motu',
        'Natural Language :: Hungarian',
        'Natural Language :: Icelandic',
        'Natural Language :: Igbo',
        'Natural Language :: Iloko',
        'Natural Language :: Indic languages',
        'Natural Language :: Indo-European languages',
        'Natural Language :: Indo-Iranian languages',
        'Natural Language :: Indonesian',
        'Natural Language :: Irish',
        'Natural Language :: Isoko',
        'Natural Language :: Italian',
        'Natural Language :: Italic languages',
        'Natural Language :: Japanese',
        'Natural Language :: Japanese',
        'Natural Language :: Kabyle',
        'Natural Language :: Kalaallisut, Greenlandic',
        'Natural Language :: Kannada',
        'Natural Language :: Kaonde',
        'Natural Language :: Khmer',
        'Natural Language :: Kinyarwanda',
        'Natural Language :: Kirundi',
        'Natural Language :: Kongo',
        'Natural Language :: Korean',
        'Natural Language :: Kwangali',
        'Natural Language :: Kwanyama, Kuanyama',
        'Natural Language :: Lao',
        'Natural Language :: Latvian',
        'Natural Language :: Lingala',
        'Natural Language :: Lithuanian',
        'Natural Language :: Lozi',
        'Natural Language :: Luba-Katanga',
        'Natural Language :: Luba-Lulua',
        'Natural Language :: Lunda',
        'Natural Language :: Lushai',
        'Natural Language :: Luvale',
        'Natural Language :: Macedonian',
        'Natural Language :: Malagasy',
        'Natural Language :: Malay',
        'Natural Language :: Malayalam',
        'Natural Language :: Malayo-Polynesian languages',
        'Natural Language :: Maltese',
        'Natural Language :: Manx',
        'Natural Language :: Marathi (Marāṭhī)',
        'Natural Language :: Marshallese',
        'Natural Language :: Mon-Khmer languages',
        'Natural Language :: Morisyen',
        'Natural Language :: Mossi',
        'Natural Language :: Multiple languages',
        'Natural Language :: Ndonga',
        'Natural Language :: Nepali',
        'Natural Language :: Niger-Kordofanian languages',
        'Natural Language :: Niuean',
        'Natural Language :: North Germanic languages',
        'Natural Language :: Northern Sotho, Pedi, Sepedi',
        'Natural Language :: Norwegian',
        'Natural Language :: Nyaneka',
        'Natural Language :: Oriya',
        'Natural Language :: Oromo',
        'Natural Language :: Pangasinan',
        'Natural Language :: Papiamento',
        'Natural Language :: Pashto, Pushto',
        'Natural Language :: Persian (Farsi)',
        'Natural Language :: Philippine languages',
        'Natural Language :: Pijin',
        'Natural Language :: Pohnpeian',
        'Natural Language :: Polish',
        'Natural Language :: Portuguese',
        'Natural Language :: Portuguese-based creoles and pidgins',
        'Natural Language :: Punjabi (Eastern)',
        'Natural Language :: Romance languages',
        'Natural Language :: Romanian',
        'Natural Language :: Rundi',
        'Natural Language :: Russian',
        'Natural Language :: Ruund',
        'Natural Language :: Salishan languages',
        'Natural Language :: Samoan',
        'Natural Language :: San Salvador Kongo',
        'Natural Language :: Sango',
        'Natural Language :: Sayula Popoluca',
        'Natural Language :: Semitic languages',
        'Natural Language :: Serbian',
        'Natural Language :: Seselwa Creole French',
        'Natural Language :: Shona',
        'Natural Language :: Sindhi',
        'Natural Language :: Sinhalese, Sinhala',
        'Natural Language :: Sino-Tibetan languages',
        'Natural Language :: Slavic languages',
        'Natural Language :: Slovak',
        'Natural Language :: Slovene',
        'Natural Language :: South Caucasian languages',
        'Natural Language :: South Slavic languages',
        'Natural Language :: Southern Sotho',
        'Natural Language :: Spanish',
        'Natural Language :: Sranan Tongo',
        'Natural Language :: Swahili',
        'Natural Language :: Swati',
        'Natural Language :: Swedish',
        'Natural Language :: Tagalog',
        'Natural Language :: Tahitian',
        'Natural Language :: Tai',
        'Natural Language :: Tamil',
        'Natural Language :: Telugu',
        'Natural Language :: Tetela',
        'Natural Language :: Tetun Dili',
        'Natural Language :: Thai',
        'Natural Language :: Tigrinya',
        'Natural Language :: Tiv',
        'Natural Language :: Tok Pisin',
        'Natural Language :: Tonga (Tonga Islands)',
        'Natural Language :: Tonga (Zambia)',
        'Natural Language :: Tsonga',
        'Natural Language :: Tswana',
        'Natural Language :: Tumbuka',
        'Natural Language :: Turkic languages',
        'Natural Language :: Turkish',
        'Natural Language :: Tuvalu',
        'Natural Language :: Twi',
        'Natural Language :: Ukrainian',
        'Natural Language :: Umbundu',
        'Natural Language :: Uralic languages',
        'Natural Language :: Urdu',
        'Natural Language :: Uyghur',
        'Natural Language :: Venda',
        'Natural Language :: Vietnamese',
        'Natural Language :: Wallisian',
        'Natural Language :: Walloon',
        'Natural Language :: Waray (Philippines)',
        'Natural Language :: Welsh',
        'Natural Language :: West Germanic languages',
        'Natural Language :: West Slavic languages',
        'Natural Language :: Western Malayo-Polynesian languages',
        'Natural Language :: Wolaitta, Wolaytta',
        'Natural Language :: Xhosa',
        'Natural Language :: Yapese',
        'Natural Language :: Yoruba',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Internationalization',
        'Topic :: Software Development :: Localization',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed'
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='NLP spark vision speech deep learning transformer tensorflow BERT GPT-2 Wav2Vec2 ViT',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=['my_module'],

    packages=find_packages(exclude=['test']),

    include_package_data=False  # Needed to install jar file

)
