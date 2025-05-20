#  Copyright 2017-2025 John Snow Labs
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
import sparknlp
from sparknlp.internal import ExtendedJavaWrapper


class Partition(ExtendedJavaWrapper):
    """
    The Partition class is a unified interface for extracting structured content from various document types
    using Spark NLP readers. It supports reading from files, URLs, in-memory strings, or byte arrays,
    and returns parsed output as a structured Spark DataFrame.

    Supported formats include plain text, HTML, Word (.doc/.docx), Excel (.xls/.xlsx), PowerPoint (.ppt/.pptx),
    email files (.eml, .msg), and PDFs.

    The class detects the appropriate reader either from the file extension or a provided MIME contentType,
    and delegates to the relevant method of SparkNLPReader. Custom behavior (like title thresholds,
    page breaks, etc.) can be configured through the params map during initialization.

    By abstracting reader initialization, type detection, and parsing logic, Partition simplifies
    document ingestion in scalable NLP pipelines.

    Parameters
    ----------
    params : dict, optional
        Parameter with custom configuration
        It includes the following parameters:
            - content_type (All): Override automatic file type detection.
            - store_content (All): Include raw file content in the output DataFrame as a separate 'content' column.
            - timeout (HTML): Timeout in seconds for fetching remote HTML content.
            - title_font_size (HTML, Excel): Minimum font size used to identify titles based on formatting.
            - include_page_breaks (Word, Excel): Whether to tag content with page break metadata.
            - group_broken_paragraphs (Text): Whether to merge broken lines into full paragraphs using heuristics.
            - title_length_size (Text): Max character length used to qualify text blocks as titles.
            - paragraph_split (Text): Regex to detect paragraph boundaries when grouping lines.
            - short_line_word_threshold (Text): Max word count for a line to be considered short.
            - threshold (Text): Ratio of empty lines used to switch between newline-based and paragraph grouping.
            - max_line_count (Text): Max lines evaluated when analyzing paragraph structure.
            - include_slide_notes (PowerPoint): Whether to include speaker notes from slides as narrative text.
            - infer_table_structure (Word, Excel, PowerPoint): Generate full HTML table structure from parsed table content.
            - append_cells (Excel): Append all rows into a single content block instead of individual elements.
            - cell_separator (Excel): String used to join cell values in a row for text output.
            - add_attachment_content (Email): Include text content of plain-text attachments in the output.
            - headers (HTML): This is used when a URL is provided, allowing you to set the necessary headers for the request`

    Example 1 (Reading Text Files)
    ----------
    txt_directory = "/content/txtfiles/reader/txt"
    partition_df = Partition(content_type = "text/plain").partition(txt_directory)
    partition_df.show()

     +--------------------+--------------------+
     |                path|                 txt|
     +--------------------+--------------------+
     |file:/content/txt...|[{Title, BIG DATA...|
     +--------------------+--------------------+

     Example 2 (Reading Image Files)
     ----------
     partition_df = Partition().partition("./email-files/test-several-attachments.eml")
     partition_df.show()

    +--------------------+--------------------+
    |                path|               email|
    +--------------------+--------------------+
    |file:/content/ema...|[{Title, Test Sev...|
    +--------------------+--------------------+

     Example 3 (Reading Webpages)
     ----------
     partition_df = Partition().partition("https://www.wikipedia.com", headers = {"Accept-Language": "es-ES"})
     partition_df.show()

    +--------------------+--------------------+
    |                 url|                html|
    +--------------------+--------------------+
    |https://www.wikip...|[{Title, Wikipedi...|
    +--------------------+--------------------+

    For more examples, please refer - examples/python/data-preprocessing/SparkNLP_Partition_Reader_Demo.ipynb

    """
    def  __init__(self, **kwargs):
        self.spark = sparknlp.start()
        params = {}
        for key, value in kwargs.items():
            try:
                params[key] = str(value)
            except Exception as e:
                raise ValueError(f"Invalid value for key '{key}': Cannot cast {type(value)} to string. Original error: {e}")

        super(Partition, self).__init__("com.johnsnowlabs.partition.Partition", params)

    """
        Takes a URL/file/directory path to read and parse it's content.
        
        Parameters
        ----------
        path : string   
            Path to a file or local directory. Supports URLs and DFS file systems like databricks, HDFS and Microsoft Fabric OneLake.
        headers: dict, optional
            If the path is a URL it sets the necessary headers for the request.
        
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with parsed content.
    """

    def partition(self, path, headers=None):
        if headers is None:
            headers = {}
        jdf = self._java_obj.partition(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    """
        Parses and reads data from multiple URL's.
        
        Parameters
        ----------
        urls : List[str] 
            list of URL's
        headers: dict, optional
            sets the necessary headers for the URL request.
        
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with parsed url content.
            
        Examples
        -------
        urls_df = Partition().partition_urls(["https://www.wikipedia.org", "https://example.com/"])
        urls_df.show()
        +--------------------+--------------------+
        |                 url|                html|
        +--------------------+--------------------+
        |https://www.wikip...|[{Title, Wikipedi...|
        |https://example.com/|[{Title, Example ...|
        +--------------------+--------------------+
        
        urls_df.printSchema()
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
    def partition_urls(self, path, headers=None):
        if headers is None:
            headers = {}
        jdf = self._java_obj.partitionUrlsJava(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe

    """
        Partitions text from a string.
        
        Parameters
        ----------
        text : string
            text data in string form 
            
        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with parsed text content.
        
        Examples
        -------
        raw_text = (
            "The big brown fox\n"
            "was walking down the lane.\n"
            "\n"
            "At the end of the lane,\n"
            "the fox met a bear."
        )
        
        text_df = Partition(group_broken_paragraphs=True).partition_text(text = raw_text)
        text_df.show()
        
        +--------------------------------------+   
        |txt                                   |   
        +--------------------------------------+   
        |[{NarrativeText, The big brown fox was|   
        +--------------------------------------+   
        
        
        text_df.printSchema()
        root
         |-- txt: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- elementType: string (nullable = true)
         |    |    |-- content: string (nullable = true)
         |    |    |-- metadata: map (nullable = true)
         |    |    |    |-- key: string
         |    |    |    |-- value: string (valueContainsNull = true)
    """
    def partition_text(self, text):
        jdf = self._java_obj.partitionText(text)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe