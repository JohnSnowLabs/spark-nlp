class ParamsGettersSetters(Params):
    getter_attrs = []

    def __init__(self):
        super(ParamsGettersSetters, self).__init__()
        for param in self.params:
            param_name = param.name
            fg_attr = "get" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            fs_attr = "set" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            # Generates getter and setter only if not exists
            try:
                getattr(self, fg_attr)
            except AttributeError:
                setattr(self, fg_attr, self.getParamValue(param_name))
            try:
                getattr(self, fs_attr)
            except AttributeError:
                setattr(self, fs_attr, self.setParamValue(param_name))

    def getParamValue(self, paramName):
        """Gets the value of a parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter
        """

        def r():
            try:
                return self.getOrDefault(paramName)
            except KeyError:
                return None

        return r

    def setParamValue(self, paramName):
        """Sets the value of a parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter
        """

        def r(v):
            self.set(self.getParam(paramName), v)
            return self

        return r

