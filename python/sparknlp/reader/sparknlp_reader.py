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

    Reading HTML

    >>> html_df = reader.html("https://www.wikipedia.org")
    >>> # Or with shorthand
    >>> import sparknlp
    >>> html_df = sparknlp.read().html("https://www.wikipedia.org")

    Reading PDF

    >>> pdf_df = reader.pdf("home/user/pdfs-directory")
    >>> # Or with shorthand
    >>> pdf_df = sparknlp.read().pdf("home/user/pdfs-directory")

    Reading Email
    
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
        >>> email_df.show()
        +---------------------------------------------------+
        |email                                              |
        +---------------------------------------------------+
        |[{Title, Email Text Attachments, {sent_to -> Danilo|
        +---------------------------------------------------+
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
        >>> doc_df.show()
        +-------------------------------------------------+
        |doc                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
        +-------------------------------------------------+
        |[{Table, Header Col 1, {}}, {Table, Header Col 2,|
        +-------------------------------------------------+

        >>> doc_df.printSchema()
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
        >>> xlsDf.show()
        +--------------------------------------------+
        |xls                                         |
        +--------------------------------------------+
        |[{Title, Financial performance, {SheetNam}}]|
        +--------------------------------------------+

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
        +-------------------------------------+
        |ppt                                  |
        +-------------------------------------+
        |[{Title, Adding a Bullet Slide, {}},]|
        +-------------------------------------+
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
        +-----------------------------------------------+
        |txt                                            |
        +-----------------------------------------------+
        |[{Title, BIG DATA ANALYTICS, {paragraph -> 0}}]|
        +-----------------------------------------------+
        """
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.txt(docPath)
        return self.getDataFrame(self.spark, jdf)

    def xml(self, docPath):
        """Reads XML files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to an XML file or a directory containing XML files.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed XML content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> xml_df = SparkNLPReader(spark).xml("home/user/xml-directory")

        You can use SparkNLP for one line of code

        >>> import sparknlp
        >>> xml_df = sparknlp.read().xml("home/user/xml-directory")
        >>> xml_df.show(truncate=False)
        +-----------------------------------------------------------+
        |xml                                                       |
        +-----------------------------------------------------------+
        |[{Title, John Smith, {elementId -> ..., tag -> title}}]   |
        +-----------------------------------------------------------+

        >>> xml_df.printSchema()
        root
         |-- path: string (nullable = true)
         |-- xml: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- elementType: string (nullable = true)
         |    |    |-- content: string (nullable = true)
         |    |    |-- metadata: map (nullable = true)
         |    |    |    |-- key: string
         |    |    |    |-- value: string (valueContainsNull = true)
        """
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.xml(docPath)
        return self.getDataFrame(self.spark, jdf)


    def md(self, filePath):
        """Reads Markdown files and returns a Spark DataFrame.

        Parameters
        ----------
        filePath : str
            Path to a Markdown file or a directory containing Markdown files.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed Markdown content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> md_df = SparkNLPReader(spark).md("home/user/markdown-directory")

        You can use SparkNLP for one line of code

        >>> import sparknlp
        >>> md_df = sparknlp.read().md("home/user/markdown-directory")
        >>> md_df.show(truncate=False)
        +-----------------------------------------------------------+
        |md                                                         |
        +-----------------------------------------------------------+
        |[{Title, Sample Markdown Document, {elementId -> ..., tag -> title}}]|
        +-----------------------------------------------------------+

        >>> md_df.printSchema()
        root
         |-- path: string (nullable = true)
         |-- md: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- elementType: string (nullable = true)
         |    |    |-- content: string (nullable = true)
         |    |    |-- metadata: map (nullable = true)
         |    |    |    |-- key: string
         |    |    |    |-- value: string (valueContainsNull = true)
        """
        if not isinstance(filePath, str):
            raise TypeError("filePath must be a string")
        jdf = self._java_obj.md(filePath)
        return self.getDataFrame(self.spark, jdf)

    def csv(self, csvPath):
        """Reads CSV files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to an CSV file or a directory containing CSV files.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed CSV content.

        Examples
        --------
        >>> from sparknlp.reader import SparkNLPReader
        >>> csv_df = SparkNLPReader(spark).csv("home/user/csv-directory")

        You can use SparkNLP for one line of code

        >>> import sparknlp
        >>> csv_df = sparknlp.read().csv("home/user/csv-directory")
        >>> csv_df.show(truncate=False)
        +-----------------------------------------------------------------------------------------------------------------------------------------+
        |csv                                                                                                                                      |
        +-----------------------------------------------------------------------------------------------------------------------------------------+
        |[{NarrativeText, Alice 100 Bob 95, {}}, {Table, <table><tr><td>Alice</td><td>100</td></tr><tr><td>Bob</td><td>95</td></tr></table>, {}}] |
        +-----------------------------------------------------------------------------------------------------------------------------------------+

        >>> csv_df.printSchema()
        root
         |-- path: string (nullable = true)
         |-- csv: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- elementType: string (nullable = true)
         |    |    |-- content: string (nullable = true)
         |    |    |-- metadata: map (nullable = true)
         |    |    |    |-- key: string
         |    |    |    |-- value: string (valueContainsNull = true)
        """
        if not isinstance(csvPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.csv(csvPath)
        return self.getDataFrame(self.spark, jdf)