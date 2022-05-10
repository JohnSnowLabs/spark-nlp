class AnnotatorModel(JavaModel, _internal.AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties,
                     _internal.ParamsGettersSetters):

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def __init__(self, classname, java_model=None):
        super(AnnotatorModel, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()
        self._setDefault(lazyAnnotator=False)

