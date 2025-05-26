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
"""Contains the Partition annotator for reading and processing various document types."""
import sparknlp
from sparknlp.internal import ExtendedJavaWrapper


class Partition(ExtendedJavaWrapper):
    """
    A unified interface for extracting structured content from various document types
    using Spark NLP readers.

    This class supports reading from files, URLs, in-memory strings, or byte arrays,
    and returns parsed output as a structured Spark DataFrame.

    Supported formats include:
    - Plain text
    - HTML
    - Word (.doc/.docx)
    - Excel (.xls/.xlsx)
    - PowerPoint (.ppt/.pptx)
    - Email files (.eml, .msg)
    - PDFs

    Parameters
    ----------
    params : dict, optional
        Configuration parameters, including:

        - content_type : str
            Override automatic file type detection.
        - store_content : bool
            Include raw file content in the output DataFrame.
        - timeout : int
            Timeout for fetching HTML content.
        - title_font_size : int
            Font size used to identify titles.
        - include_page_breaks : bool
            Tag content with page break metadata.
        - group_broken_paragraphs : bool
            Merge broken lines into full paragraphs.
        - title_length_size : int
            Max character length to qualify as title.
        - paragraph_split : str
            Regex to detect paragraph boundaries.
        - short_line_word_threshold : int
            Max words in a line to be considered short.
        - threshold : float
            Ratio of empty lines for switching grouping.
        - max_line_count : int
            Max lines evaluated in paragraph analysis.
        - include_slide_notes : bool
            Include speaker notes in output.
        - infer_table_structure : bool
            Generate HTML table structure.
        - append_cells : bool
            Merge Excel rows into one block.
        - cell_separator : str
            Join cell values in a row.
        - add_attachment_content : bool
            Include text of plain-text attachments.
        - headers : dict
            Request headers when using URLs.

    Examples
    --------

    Reading Text Files

    >>> txt_directory = "/content/txtfiles/reader/txt"
    >>> partition_df = Partition(content_type="text/plain").partition(txt_directory)
    >>> partition_df.show()
    >>> partition_df = Partition().partition("./email-files/test-several-attachments.eml")
    >>> partition_df.show()
    >>> partition_df = Partition().partition(
    ...     "https://www.wikipedia.com",
    ...     headers={"Accept-Language": "es-ES"}
    ... )
    >>> partition_df.show()
    +--------------------+--------------------+
    |                path|                 txt|
    +--------------------+--------------------+
    |file:/content/txt...|[{Title, BIG DATA...|
    +--------------------+--------------------+

    Reading Email Files

    >>> partition_df = Partition().partition("./email-files/test-several-attachments.eml")
    >>> partition_df.show()
    +--------------------+--------------------+
    |                path|               email|
    +--------------------+--------------------+
    |file:/content/ema...|[{Title, Test Sev...|
    +--------------------+--------------------+

    Reading Webpages

    >>> partition_df = Partition().partition("https://www.wikipedia.com", headers = {"Accept-Language": "es-ES"})
    >>> partition_df.show()
    +--------------------+--------------------+
    |                 url|                html|
    +--------------------+--------------------+
    |https://www.wikip...|[{Title, Wikipedi...|
    +--------------------+--------------------+

    For more examples, refer to:
    `examples/python/data-preprocessing/SparkNLP_Partition_Reader_Demo.ipynb`
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


    def partition(self, path, headers=None):
        """
        Reads and parses content from a URL, file, or directory path.

        Parameters
        ----------
        path : str
            Path to file or directory. URLs and DFS are supported.
        headers : dict, optional
            Headers for URL requests.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with parsed content.
    """
        if headers is None:
            headers = {}
        jdf = self._java_obj.partition(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe


    def partition_urls(self, path, headers=None):
        """
        Reads and parses content from multiple URLs.

        Parameters
        ----------
        path : list[str]
            List of URLs.
        headers : dict, optional
            Request headers for URLs.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with parsed URL content.

        Examples
        --------
        >>> urls_df = Partition().partition_urls([
        ...     "https://www.wikipedia.org", "https://example.com/"
        ... ])
        >>> urls_df.show()
        +--------------------+--------------------+
        |                 url|                html|
        +--------------------+--------------------+
        |https://www.wikip...|[{Title, Wikipedi...|
        |https://example.com/|[{Title, Example ...|
        +--------------------+--------------------+

        >>> urls_df.printSchema()
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
        if headers is None:
            headers = {}
        jdf = self._java_obj.partitionUrlsJava(path, headers)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe


    def partition_text(self, text):
        """
        Parses content from a raw text string.

        Parameters
        ----------
        text : str
            Raw text input.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with parsed text.

        Examples
        --------
        >>> raw_text = (
        ...     "The big brown fox\\n"
        ...     "was walking down the lane.\\n"
        ...     "\\n"
        ...     "At the end of the lane,\\n"
        ...     "the fox met a bear."
        ... )
        >>> text_df = Partition(group_broken_paragraphs=True).partition_text(text=raw_text)
        >>> text_df.show()
        +--------------------------------------+
        |txt                                   |
        +--------------------------------------+
        |[{NarrativeText, The big brown fox was|
        +--------------------------------------+
        >>> text_df.printSchema()
        root
         |-- txt: array (nullable = true)
         |    |-- element: struct (containsNull = true)
         |    |    |-- elementType: string (nullable = true)
         |    |    |-- content: string (nullable = true)
         |    |    |-- metadata: map (nullable = true)
         |    |    |    |-- key: string
         |    |    |    |-- value: string (valueContainsNull = true)
        """
        jdf = self._java_obj.partitionText(text)
        dataframe = self.getDataFrame(self.spark, jdf)
        return dataframe