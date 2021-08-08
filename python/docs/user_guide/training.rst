*****************************
Loading datasets for training
*****************************

There are several helper classes in Spark NLP to make training your own models easier.

POS Dataset
===========

In order to train a Part of Speech Tagger annotator
(:class:`PerceptronApproach <sparknlp.annotator.PerceptronApproach>`), we need to
get corpus data as a Spark dataframe. :class:`POS <sparknlp.training.POS>` reads a plain text file
and transforms it to a Spark dataset.

**Input File Format**::

    A|DT few|JJ months|NNS ago|RB you|PRP received|VBD a|DT letter|NN

**Example**

>>> from sparknlp.training import POS
>>> train_pos = POS().readDataset(spark, "./src/main/resources/anc-pos-corpus")

CoNLL Dataset
=============

In order to train a :class:`sparknlp.annotator.NerDLApproach` annotator, we need to get
`CoNLL 2003 <https://www.clips.uantwerpen.be/conll2003/ner/>`_ format data
as a Spark dataframe. :class:`sparknlp.training.CoNLL` reads a plain text file and transforms it to a Spark dataset.

**Input File Format**::

    -DOCSTART- -X- -X- O

    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    call NN I-NP O
    to TO B-VP O
    boycott VB I-VP O
    British JJ B-NP B-MISC
    lamb NN I-NP O
    . . O O

**Example**

>>> from sparknlp.training import CoNLL
>>> training_conll = CoNLL().readDataset(spark, "./src/main/resources/conll2003/eng.train")

CoNLLU Dataset
=============

In order to train a :class:`DependencyParserApproach <sparknlp.annotator.DependencyParserApproach>` annotator, we need to get
`CoNLL-U <https://universaldependencies.org/format.html>`_ format data
as a Spark dataframe. :class:`CoNLLU <sparknlp.training.CoNLLU>` reads a plain text file and transforms it to a Spark dataset.

**Input File Format**::

    -DOCSTART- -X- -X- O

    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    call NN I-NP O
    to TO B-VP O
    boycott VB I-VP O
    British JJ B-NP B-MISC
    lamb NN I-NP O
    . . O O

**Example**

>>> from sparknlp.training import CoNLLU
>>> conlluFile = "src/test/resources/conllu/en.test.conllu"
>>> conllDataSet = CoNLLU(False).readDataset(spark, conlluFile)

Spell Checkers Dataset
======================
In order to train a :class:`NorvigSweetingApproach <sparknlp.annotator.NorvigSweetingApproach>` or
:class:`SymmetricDeleteApproach <sparknlp.annotator.SymmetricDeleteApproach>`, we need to get corpus data as a spark
dataframe. We can read any plain text file and transform it to a Spark dataset.

**Example**

>>> train_corpus = spark.read.text("./sherlockholmes.txt").withColumnRenamed("value", "text")


PubTator Dataset
================
The PubTator format includes medical papers’ titles, abstracts, and tagged chunks
(see PubTator Docs and MedMentions Docs for more information).
We can create a Spark DataFrame from a PubTator text file with :class:`PubTator <sparknlp.training.PubTator>`.

**Input File Format**::

    25763772	0	5	DCTN4	T116,T123	C4308010
    25763772	23	63	chronic Pseudomonas aeruginosa infection	T047	C0854135
    25763772	67	82	cystic fibrosis	T047	C0010674
    25763772	83	120	Pseudomonas aeruginosa (Pa) infection	T047	C0854135
    25763772	124	139	cystic fibrosis	T047	C0010674

**Example**

>>> from sparknlp.training import PubTator
>>> trainingPubTatorDF = PubTator.readDataset(spark, "./src/test/resources/corpus_pubtator.txt")
