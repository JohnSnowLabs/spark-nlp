#  Copyright 2017-2024 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from sparknlp.internal import ExtendedJavaWrapper


class SparkNLPReader(ExtendedJavaWrapper):
    """Instantiates class to read HTML, email, and document files.

    Two types of input paths are supported:

    - `htmlPath`: A path to a directory of HTML files or a single HTML file (e.g., `"path/html/files"`).
    - `url`: A single URL or a set of URLs (e.g., `"https://www.wikipedia.org"`).

    Parameters
    ----------
    params : spark
        Spark session
    params : dict, optional
        Parameter with custom configuration

    Examples
    --------
    >>> from sparknlp.reader import SparkNLPReader
    >>> html_df = SparkNLPReader().html(spark, "https://www.wikipedia.org")

    You can use SparkNLP for one line of code
    >>> import sparknlp
    >>> html_df = sparknlp.read().html("https://www.wikipedia.org")
    >>> html_df.show(truncate=False)

    +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |url                 |html                                                                                                                                                                                                                                                                                                                            |
    +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |https://example.com/|[{Title, Example Domain, {pageNumber -> 1}}, {NarrativeText, 0, This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission., {pageNumber -> 1}}, {NarrativeText, 0, More information... More information..., {pageNumber -> 1}}]   |
    +--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    >>> html_df.printSchema()

    root
     |-- url: string (nullable = true)
     |-- html: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- elementType: string (nullable = true)
     |    |    |-- content: string (nullable = true)
     |    |    |-- metadata: map (nullable = true)
     |    |    |    |-- key: string
     |    |    |    |-- value: string (valueContainsNull = true)



    Instantiates class to read email files.

    emailPath: this is a path to a directory of HTML files or a path to an HTML file E.g.
    "path/html/emails"

    Examples
    --------
    >>> from sparknlp.reader import SparkNLPReader
    >>> email_df = SparkNLPReader().email(spark, "home/user/emails-directory")

    You can use SparkNLP for one line of code
    >>> import sparknlp
    >>> email_df = sparknlp.read().email("home/user/emails-directory")
    >>> email_df.show(truncate=False)
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |email                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[{Title, Email Text Attachments, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>}}, {NarrativeText, Email  test with two text attachments\r\n\r\nCheers,\r\n\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {NarrativeText, <html>\r\n<head>\r\n<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">\r\n<style type="text/css" style="display:none;"> P {margin-top:0;margin-bottom:0;} </style>\r\n</head>\r\n<body dir="ltr">\r\n<span style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">Email&nbsp; test with two text attachments</span>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\nCheers,</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n</body>\r\n</html>\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/html}}, {Attachment, filename.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename.txt"}}, {NarrativeText, This is the content of the file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {Attachment, filename2.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename2.txt"}}, {NarrativeText, This is an additional content file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}]|
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    email_df.printSchema()
    root
     |-- path: string (nullable = true)
     |-- content: array (nullable = true)
     |-- email: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- elementType: string (nullable = true)
     |    |    |-- content: string (nullable = true)
     |    |    |-- metadata: map (nullable = true)
     |    |    |    |-- key: string
     |    |    |    |-- value: string (valueContainsNull = true)


    Instantiates class to read PDF files.

    pdfPath: this is a path to a directory of PDF files or a path to an PDF file E.g.
    "path/pdfs/"

    Examples
    --------
    >>> from sparknlp.reader import SparkNLPReader
    >>> pdf_df = SparkNLPReader().pdf(spark, "home/user/pdfs-directory")

    You can use SparkNLP for one line of code
    >>> import sparknlp
    >>> pdf_df = sparknlp.read().pdf("home/user/pdfs-directory")
    >>> pdf_df.show(truncate=False)

    +--------------------+--------------------+------+--------------------+----------------+---------------+--------------------+---------+-------+
    |                path|    modificationTime|length|                text|height_dimension|width_dimension|             content|exception|pagenum|
    +--------------------+--------------------+------+--------------------+----------------+---------------+--------------------+---------+-------+
    |file:/content/pdf...|2025-01-15 20:48:...| 25803|This is a Title \...|             842|            596|[25 50 44 46 2D 3...|     NULL|      0|
    |file:/content/pdf...|2025-01-15 20:48:...|  9487|This is a page.\n...|             841|            595|[25 50 44 46 2D 3...|     NULL|      0|
    +--------------------+--------------------+------+--------------------+----------------+---------------+--------------------+---------+-------+

    pdf_df.printSchema()
    root
     |-- path: string (nullable = true)
     |-- modificationTime: timestamp (nullable = true)
     |-- length: long (nullable = true)
     |-- text: string (nullable = true)
     |-- height_dimension: integer (nullable = true)
     |-- width_dimension: integer (nullable = true)
     |-- content: binary (nullable = true)
     |-- exception: string (nullable = true)
     |-- pagenum: integer (nullable = true)
    """

    def __init__(self, spark, params=None):
        if params is None:
            params = {}
        super(SparkNLPReader, self).__init__("com.johnsnowlabs.reader.SparkNLPReader", params)
        self.spark = spark

    def html(self, htmlPath):
        if not isinstance(htmlPath, (str, list)) or (isinstance(htmlPath, list) and not all(isinstance(item, str) for item in htmlPath)):
            raise TypeError("htmlPath must be a string or a list of strings")
        jdf = self._java_obj.html(htmlPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def email(self, filePath):
        if not isinstance(filePath, str):
            raise TypeError("filePath must be a string")
        jdf = self._java_obj.email(filePath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def doc(self, docPath):
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.doc(docPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def pdf(self, pdfPath):
        if not isinstance(pdfPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.pdf(pdfPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe