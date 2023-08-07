#  Copyright 2017-2023 John Snow Labs
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
"""Contains classes for OpenAICompletion."""
from sparknlp.common import *

"""Transformer that makes a request for OpenAI Completion API for each executor.
   
   See OpenAI API Doc: https://platform.openai.com/docs/api-reference/completions/create for reference 
    
   ====================== ======================
   Input Annotation types Output Annotation type
   ====================== ======================
   ``DOCUMENT``           ``DOCUMENT``
   ====================== ======================

   Parameters
   ----------
   model
       ID of the OpenAI model to use
   suffix
        The suffix that comes after a completion of inserted text
   maxTokens
        The maximum number of tokens to generate in the completion.
   temperature
        What sampling temperature to use, between 0 and 2 
   topP
        An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass
   numberOfCompletions
        How many completions to generate for each prompt.
   logprobs
        Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
   echo          
       Echo back the prompt in addition to the completion
   stop
      Up to 4 sequences where the API will stop generating further tokens.
   presencePenalty   
      Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
   frequencyPenalty
      Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
   bestOf      
      Generates best_of completions server-side and returns the `best` (the one with the highest log probability per token).
   logitBias
      Modify the likelihood of specified tokens appearing in the completion.
   user
      A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.      
        
   Examples
   --------
   >>> import sparknlp
   >>> from sparknlp.base import *
   >>> from sparknlp.annotator import *
   >>> from sparknlp.common import *
   >>> from pyspark.ml import Pipeline

   >>> documentAssembler = DocumentAssembler() \\
   ...     .setInputCol("text") \\
   ...     .setOutputCol("document")
   >>> openai_completion = OpenAICompletion() \\
   ...     .setInputCols("document") \\
   ...     .setOutputCol("completion") \\
   ...     .setModel("text-davinci-003") \\
   ...     .setMaxTokens(100)
   >>> pipeline = Pipeline().setStages([
   ...     documentAssembler,
   ...     openai_completion
   ... ])
   >>> empty_df = spark.createDataFrame([[""]], ["text"])
   >>> sample_text= [["Generate a restaurant review."], ["Write a review for a local eatery."], ["Create a JSON with a review of a dining experience."]]
   >>> sample_df = spark.createDataFrame(sample_text).toDF("text")
   >>> sample_df.show()
   +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |completion                                                                                                                                                                                                                                                                                        |
   +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |[{document, 0, 258, \n\nI had the pleasure of dining at La Fiorita recently, and it was a truly delightful experience! The menu boasted a wonderful selection of classic Italian dishes, all exquisitely prepared and presented. The service staff was friendly and attentive and really, {}, []}]|
   |[{document, 0, 227, \n\nI recently visited Barbecue Joe's for dinner and it was amazing! The menu had so many items to choose from including pulled pork, smoked turkey, brisket, pork ribs, and sandwiches. I opted for the pulled pork sandwich and let, {}, []}]                               |
   |[{document, 0, 172, \n\n{ \n   "review": { \n      "overallRating": 4, \n      "reviewBody": "I enjoyed my meal at this restaurant. The food was flavourful, well-prepared and beautifully presented., {}, []}]                                                                                   |
   +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   """


class OpenAICompletion(AnnotatorModel):

    name = "OpenAICompletion"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    model = Param(Params._dummy(),
                  "model",
                  "ID of the OpenAI model to use",
                  typeConverter=TypeConverters.toString)

    suffix = Param(Params._dummy(),
                   "suffix",
                   "The suffix that comes after a completion of inserted text.",
                   typeConverter=TypeConverters.toString)

    maxTokens = Param(Params._dummy(),
                      "maxTokens",
                      "The maximum number of tokens to generate in the completion.",
                      typeConverter=TypeConverters.toInt)

    temperature = Param(Params._dummy(),
                        "temperature",
                        "What sampling temperature to use, between 0 and 2",
                        typeConverter=TypeConverters.toFloat)

    topP = Param(Params._dummy(),
                      "topP",
                      "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass",
                      typeConverter=TypeConverters.toFloat)

    numberOfCompletions = Param(Params._dummy(),
                                   "numberOfCompletions",
                                   "How many completions to generate for each prompt.",
                                   typeConverter=TypeConverters.toInt)

    logprobs = Param(Params._dummy(),
                         "logprobs",
                         "Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.",
                         typeConverter=TypeConverters.toInt)

    echo = Param(Params._dummy(),
                        "echo",
                        "Echo back the prompt in addition to the completion",
                        typeConverter=TypeConverters.toBoolean)

    stop = Param(Params._dummy(),
                  "stop",
                  "Up to 4 sequences where the API will stop generating further tokens.",
                  typeConverter=TypeConverters.toListString)

    presencePenalty = Param(Params._dummy(),
                                 "presencePenalty",
                                 "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
                                 typeConverter=TypeConverters.toFloat)

    frequencyPenalty = Param(Params._dummy(),
                                  "frequencyPenalty",
                                  "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
                                  typeConverter=TypeConverters.toFloat)

    bestOf = Param(Params._dummy(),
                      "bestOf",
                      "Generates best_of completions server-side and returns the `best` (the one with the highest log probability per token).",
                      typeConverter=TypeConverters.toInt)

    logitBias = Param(Params._dummy(),
                  "logitBias",
                  "Modify the likelihood of specified tokens appearing in the completion.",
                  typeConverter=TypeConverters.identity)

    user = Param(Params._dummy(),
                 "user",
                 "A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.",
                 typeConverter=TypeConverters.toString)

    def setModel(self, value):
        """Sets model ID of the OpenAI model to use

        Parameters
        ----------
        value : str
           ID of the OpenAI model to use
        """
        return self._set(model=value)

    def setSuffix(self, value):
        """Sets the suffix that comes after a completion of inserted text.

        Parameters
        ----------
        value : str
           The suffix that comes after a completion of inserted text.
        """
        return self._set(suffix=value)

    def setMaxTokens(self, value):
        """Sets the maximum number of tokens to generate in the completion.
           The token count of your prompt plus max_tokens cannot exceed the model's context length.

        Parameters
        ----------
        value : int
           The maximum number of tokens to generate in the completion.
        """
        return self._set(maxTokens=value)

    def setTemperature(self, value):
        """Sets what sampling temperature to use, between 0 and 2.
        Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or top_p but not both.

        Parameters
        ----------
        value : float
           What sampling temperature to use, between 0 and 2
        """
        return self._set(temperature=value)

    def setTopP(self, value):
        """Sets An alternative to sampling with temperature, called nucleus sampling.
        Where the model considers the results of the tokens with top_p probability mass.
        So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or temperature but not both.

        Parameters
        ----------
        value : float
           An alternative to sampling with temperature, called nucleus sampling, where the model considers the results
           of the tokens with top_p probability mass
        """
        return self._set(topP=value)

    def setNumberOfCompletions(self, value):
        """Sets how many completions to generate for each prompt.

        Note: Because this parameter generates many completions, it can quickly consume your token quota.
        Use carefully and ensure that you have reasonable settings for max_tokens and stop.

        Parameters
        ----------
        value : int
           How many completions to generate for each prompt.
        """
        return self._set(numberOfCompletions=value)

    def setLogprobs(self, value):
        """Sets the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
        For example, if logprobs is 5, the API will return a list of the 5 most likely tokens.
        The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
        The maximum value for logprobs is 5.

        Parameters
        ----------
        value : int
           How many completions to generate for each prompt.
        """
        return self._set(logprobs=value)

    def setEcho(self, value):
        """Sets echo back the prompt in addition to the completion

        Parameters
        ----------
        value : str
           Echo back the prompt in addition to the completion
        """
        return self._set(echo=value)

    def setStop(self, value):
        """Sets Up to 4 sequences where the API will stop generating further tokens.
        The returned text will not contain the stop sequence.

        Parameters
        ----------
        value : List[str]
           Up to 4 sequences where the API will stop generating further tokens.
        """
        return self._set(stop=value)

    def setPresencePenalty(self, value):
        """Sets values to penalize new tokens based on whether they appear in the text so far,
        increasing the model's likelihood to talk about new topics.

        Parameters
        ----------
        value : int
           Number between -2.0 and 2.0.
           Positive values penalize new tokens based on whether they appear in the text so far,
           increasing the model's likelihood to talk about new topics.
        """
        return self._set(presencePenalty=value)

    def setFrequencyPenalty(self, value):
        """Sets values to penalize new tokens based on their existing frequency in the text so far,
           decreasing the model's likelihood to repeat the same line verbatim.

        Parameters
        ----------
        value : int
           Number between -2.0 and 2.0.
           Positive values penalize new tokens based on their existing frequency in the text so far,
           decreasing the model's likelihood to repeat the same line verbatim.
        """
        return self._set(frequencyPenalty=value)

    def setBestOf(self, value):
        """Sets best_of completions server-side and returns the "best" (the one with the highest log probability per token)

        Parameters
        ----------
        value : int
           Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token).
           Results cannot be streamed.
           When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.
           Note: Because this parameter generates many completions, it can quickly consume your token quota.
           Use carefully and ensure that you have reasonable settings for max_tokens and stop.
        """
        return self._set(bestOf=value)

    def setLogitBias(self, value):
        """Sets the likelihood of specified tokens appearing in the completion

        Parameters
        ----------
        value : dict
           Modify the likelihood of specified tokens appearing in the completion.
            Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated
            bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
            As an example, you can pass {"50256": -100} to prevent the <|endoftext|> token from being generated.
        """
        return self._set(logitBias=value)

    def setUser(self, value):
        """Sets a unique identifier representing your end-user

        Parameters
        ----------
        value : str
           A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        """
        return self._set(user=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.ml.ai.OpenAICompletion", java_model=None):
        super(OpenAICompletion, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            maxTokens=16,
            temperature=1,
            topP=1,
            numberOfCompletions=1,
            echo=False,
            presencePenalty=0,
            frequencyPenalty=0,
            bestOf=1
        )
