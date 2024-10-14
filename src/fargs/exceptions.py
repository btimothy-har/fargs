class FargsError(Exception):
    pass


class FargsLLMError(FargsError):
    pass


class FargsNoResponseError(FargsLLMError):
    pass


class FargsExtractionError(FargsError):
    pass
