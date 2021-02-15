class TextInput:
    def load_text_file(self, filepath):
        f = open(filepath, "r")
        return f.read()