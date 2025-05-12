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
    """Instantiates class to read documents in various formats.

    Parameters
    ----------
    params : spark
        Spark session
    params : dict, optional
        Parameter with custom configuration

    Notes
    -----
    This class can read HTML, email, PDF, MS Word, Excel, PowerPoint, and text files.

    Examples
    --------
    >>> from sparknlp.reader import SparkNLPReader
    >>> reader = SparkNLPReader(spark)

    # Reading HTML
    >>> html_df = reader.html("https://www.wikipedia.org")
    >>> # Or with shorthand
    >>> import sparknlp
    >>> html_df = sparknlp.read().html("https://www.wikipedia.org")

    # Reading PDF
    >>> pdf_df = reader.pdf("home/user/pdfs-directory")
    >>> # Or with shorthand
    >>> pdf_df = sparknlp.read().pdf("home/user/pdfs-directory")

    # Reading Email
    >>> email_df = reader.email("home/user/emails-directory")
    >>> # Or with shorthand
    >>> email_df = sparknlp.read().email("home/user/emails-directory")
    """

    def __init__(self, spark, params=None, headers=None):
        if params is None:
            params = {}
        super(SparkNLPReader, self).__init__("com.johnsnowlabs.reader.SparkNLPReader", params, headers)
        self.spark = spark

    def html(self, htmlPath):
        """Reads HTML files or URLs and returns a Spark DataFrame.

        Parameters
        ----------
        htmlPath : str or list of str
            Path(s) to HTML file(s) or a list of URLs.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing the parsed HTML content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> html_df = SparkNLPReader().html("https://www.wikipedia.org")

        You can also use SparkNLP to simplify the process:

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
        """
        if not isinstance(htmlPath, (str, list)) or (isinstance(htmlPath, list) and not all(isinstance(item, str) for item in htmlPath)):
            raise TypeError("htmlPath must be a string or a list of strings")
        jdf = self._java_obj.html(htmlPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def email(self, filePath):
        """Reads email files and returns a Spark DataFrame.

        Parameters
        ----------
        filePath : str
            Path to an email file or a directory containing emails.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed email data.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> email_df = SparkNLPReader(spark).email("home/user/emails-directory")

        You can also use SparkNLP to simplify the process:

        >>> import sparknlp
        >>> email_df = sparknlp.read().email("home/user/emails-directory")
        >>> email_df.show(truncate=False)

        +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |email                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
        +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |[{Title, Email Text Attachments, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>}}, {NarrativeText, Email  test with two text attachments\r\n\r\nCheers,\r\n\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {NarrativeText, <html>\r\n<head>\r\n<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">\r\n<style type="text/css" style="display:none;"> P {margin-top:0;margin-bottom:0;} </style>\r\n</head>\r\n<body dir="ltr">\r\n<span style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">Email&nbsp; test with two text attachments</span>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\nCheers,</div>\r\n<div class="elementToProof" style="font-family: Aptos, Aptos_EmbeddedFont, Aptos_MSFontService, Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">\r\n<br>\r\n</div>\r\n</body>\r\n</html>\r\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/html}}, {Attachment, filename.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename.txt"}}, {NarrativeText, This is the content of the file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}, {Attachment, filename2.txt, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, contentType -> text/plain; name="filename2.txt"}}, {NarrativeText, This is an additional content file.\n, {sent_to -> Danilo Burbano <danilo@johnsnowlabs.com>, sent_from -> Danilo Burbano <danilo@johnsnowlabs.com>, mimeType -> text/plain}}]|
        +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        >>> email_df.printSchema()
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

        """
        if not isinstance(filePath, str):
            raise TypeError("filePath must be a string")
        jdf = self._java_obj.email(filePath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def doc(self, docPath):
        """Reads word document files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to a word document file.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed document content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> doc_df = SparkNLPReader().doc(spark, "home/user/word-directory")

        You can use SparkNLP for one line of code
        >>> import sparknlp
        >>> doc_df = sparknlp.read().doc("home/user/word-directory")
        >>> doc_df.show(truncate=False)

        +----------------------------------------------------------------------------------------------------------------------------------------------------+
        |doc                                                                                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
        +----------------------------------------------------------------------------------------------------------------------------------------------------+
        |[{Table, Header Col 1, {}}, {Table, Header Col 2, {}}, {Table, Lorem ipsum, {}}, {Table, A Link example, {}}, {NarrativeText, Dolor sit amet, {}}]  |
        +----------------------------------------------------------------------------------------------------------------------------------------------------+
        >>> docsDf.printSchema()
        root
         |-- path: string (nullable = true)
         |-- content: array (nullable = true)
         |-- doc: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- elementType: string (nullable = true)
         |    |    |-- content: string (nullable = true)
         |    |    |-- metadata: map (nullable = true)
         |    |    |    |-- key: string
         |    |    |    |-- value: string (valueContainsNull = true)

        """
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

    def xls(self, docPath):
        """Reads excel document files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to an excel document file.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed document content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> xlsDf = SparkNLPReader().xls(spark, "home/user/excel-directory")

        You can use SparkNLP for one line of code
        >>> import sparknlp
        >>> xlsDf = sparknlp.read().xls("home/user/excel-directory")
        >>> xlsDf.show(truncate=False)

        +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |xls                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
        +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |[{Title, Financial performance, {SheetName -> Index}}, {Title, Topic\tPeriod\t\t\tPage, {SheetName -> Index}}, {NarrativeText, Quarterly revenue\tNine quarters to 30 June 2023\t\t\t1.0, {SheetName -> Index}}, {NarrativeText, Group financial performance\tFY 22\tFY 23\t\t2.0, {SheetName -> Index}}, {NarrativeText, Segmental results\tFY 22\tFY 23\t\t3.0, {SheetName -> Index}}, {NarrativeText, Segmental analysis\tFY 22\tFY 23\t\t4.0, {SheetName -> Index}}, {NarrativeText, Cash flow\tFY 22\tFY 23\t\t5.0, {SheetName -> Index}}, {Title, Operational metrics, {SheetName -> Index}}, {Title, Topic\tPeriod\t\t\tPage, {SheetName -> Index}}, {NarrativeText, Mobile customers\tNine quarters to 30 June 2023\t\t\t6.0, {SheetName -> Index}}, {NarrativeText, Fixed broadband customers\tNine quarters to 30 June 2023\t\t\t7.0, {SheetName -> Index}}, {NarrativeText, Marketable homes passed\tNine quarters to 30 June 2023\t\t\t8.0, {SheetName -> Index}}, {NarrativeText, TV customers\tNine quarters to 30 June 2023\t\t\t9.0, {SheetName -> Index}}, {NarrativeText, Converged customers\tNine quarters to 30 June 2023\t\t\t10.0, {SheetName -> Index}}, {NarrativeText, Mobile churn\tNine quarters to 30 June 2023\t\t\t11.0, {SheetName -> Index}}, {NarrativeText, Mobile data usage\tNine quarters to 30 June 2023\t\t\t12.0, {SheetName -> Index}}, {NarrativeText, Mobile ARPU\tNine quarters to 30 June 2023\t\t\t13.0, {SheetName -> Index}}, {Title, Other, {SheetName -> Index}}, {Title, Topic\tPeriod\t\t\tPage, {SheetName -> Index}}, {NarrativeText, Average foreign exchange rates\tNine quarters to 30 June 2023\t\t\t14.0, {SheetName -> Index}}, {NarrativeText, Guidance rates\tFY 23/24\t\t\t14.0, {SheetName -> Index}}]|
        +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

       >>> xlsDf.printSchema()
       root
        |-- path: string (nullable = true)
        |-- content: binary (nullable = true)
        |-- xls: array (nullable = true)
        |    |-- element: struct (containsNull = true)
        |    |    |-- elementType: string (nullable = true)
        |    |    |-- content: string (nullable = true)
        |    |    |-- metadata: map (nullable = true)
        |    |    |    |-- key: string
        |    |    |    |-- value: string (valueContainsNull = true)
       """
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.xls(docPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def ppt(self, docPath):
        """
        Reads power point document files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to an power point document file.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed document content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> pptDf = SparkNLPReader().ppt(spark, "home/user/powerpoint-directory")

        You can use SparkNLP for one line of code
        >>> import sparknlp
        >>> pptDf = sparknlp.read().ppt("home/user/powerpoint-directory")
        >>> pptDf.show(truncate=False)
        +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |ppt                                                                                                                                                                                                                                                                                                                      |
        +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |[{Title, Adding a Bullet Slide, {}}, {ListItem, • Find the bullet slide layout, {}}, {ListItem, – Use _TextFrame.text for first bullet, {}}, {ListItem, • Use _TextFrame.add_paragraph() for subsequent bullets, {}}, {NarrativeText, Here is a lot of text!, {}}, {NarrativeText, Here is some text in a text box!, {}}]|
        +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        """
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.ppt(docPath)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    def txt(self, docPath):
        """Reads TXT files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to a TXT file.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed document content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> txtDf = SparkNLPReader().txt(spark, "home/user/txt/files")

        You can use SparkNLP for one line of code
        >>> import sparknlp
        >>> txtDf = sparknlp.read().txt("home/user/txt/files")
        >>> txtDf.show(truncate=False)
        +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |txt                                                                                                                                                                                                                                                                                                                                                                                                                                        |
        +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        |[{Title, BIG DATA ANALYTICS, {paragraph -> 0}}, {NarrativeText, Apache Spark is a fast and general-purpose cluster computing system.\nIt provides high-level APIs in Java, Scala, Python, and R., {paragraph -> 0}}, {Title, MACHINE LEARNING, {paragraph -> 1}}, {NarrativeText, Spark's MLlib provides scalable machine learning algorithms.\nIt includes tools for classification, regression, clustering, and more., {paragraph -> 1}}]|
        +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
        """
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.txt(docPath)
        return self.getDataFrame(self.spark, jdf)