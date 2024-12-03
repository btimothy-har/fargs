class FargsError(Exception):
    """Base exception class for Fargs-related errors."""

    pass


class FargsLLMError(FargsError):
    """Exception raised for errors encountered when invoking a Language Model (LLM)."""

    def __init__(self):
        super().__init__("Error invoking LLM.")


class FargsNoResponseError(FargsLLMError):
    """Exception raised when the Language Model returns an empty response."""

    def __init__(self):
        super().__init__("LLM returned an empty response.")


class FargsExtractionError(FargsError):
    """
    Exception raised when there is an error extracting output from the Language Model.
    """

    def __init__(self, message: str):
        super().__init__(f"Error extracting output from LLM: {message}.")
