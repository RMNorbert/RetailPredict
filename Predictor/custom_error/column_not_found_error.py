class ColumnNotFoundError(ValueError):
    def __init__(self, message="None of the specified columns are present in the DataFrame"):
        self.message = message
        super().__init__(self.message)
