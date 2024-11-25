class Loop_detected(Exception):
    def __init__(self):
        super().__init__("Loop detected")