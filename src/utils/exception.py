# src/utils/exception.py

class CustomException(Exception):
    """
    Simple custom exception wrapper used across the project.

    Usage:
        raise CustomException("Something went wrong", errors={"cause": "missing column"})
    """

    def __init__(self, message: str, errors: dict | None = None):
        super().__init__(message)
        self.errors = errors

    def __str__(self) -> str:
        base = super().__str__()
        if self.errors:
            return f"{base} | Errors: {self.errors}"
        return base


# simple self-test (won't run when imported)
if __name__ == "__main__":
    print("🔹 Testing CustomException...")
    try:
        raise CustomException("Test exception", errors={"reason": "unit-test"})
    except CustomException as e:
        print("Caught CustomException:", e)
