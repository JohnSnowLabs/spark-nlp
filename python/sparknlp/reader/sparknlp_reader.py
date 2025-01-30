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
    spark : SparkSession
        The active Spark session.
    params : dict, optional
        A dictionary with custom configurations.
    """

    def __init__(self, spark, params=None):
        if params is None:
            params = {}
        super(SparkNLPReader, self).__init__("com.johnsnowlabs.reader.SparkNLPReader", params)
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
        >>> html_df = SparkNLPReader(spark).html("https://www.wikipedia.org")

        You can also use SparkNLP to simplify the process:

        >>> import sparknlp
        >>> html_df = sparknlp.read().html("https://www.wikipedia.org")
        >>> html_df.show(truncate=False)
        """
        if not isinstance(htmlPath, (str, list)) or (isinstance(htmlPath, list) and not all(isinstance(item, str) for item in htmlPath)):
            raise TypeError("htmlPath must be a string or a list of strings")
        jdf = self._java_obj.html(htmlPath)
        return self.getDataFrame(self.spark, jdf)

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

        Using SparkNLP:

        >>> import sparknlp
        >>> email_df = sparknlp.read().email("home/user/emails-directory")
        >>> email_df.show(truncate=False)
        """
        if not isinstance(filePath, str):
            raise TypeError("filePath must be a string")
        jdf = self._java_obj.email(filePath)
        return self.getDataFrame(self.spark, jdf)

    def doc(self, docPath):
        """Reads document files and returns a Spark DataFrame.

        Parameters
        ----------
        docPath : str
            Path to a document file.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame containing parsed document content.
        """
        if not isinstance(docPath, str):
            raise TypeError("docPath must be a string")
        jdf = self._java_obj.doc(docPath)
        return self.getDataFrame(self.spark, jdf)