from abc import abstractmethod


class SIV:
    def __init__(self, input_dir, query_identifier, candidate_identifier, language='en'):
        self.input_dir = input_dir
        self.query_identifier = query_identifier
        self.candidate_identifier = candidate_identifier
        self.language = language

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def generate_sivs(self, input_dir, output_dir, run_id):
        pass
