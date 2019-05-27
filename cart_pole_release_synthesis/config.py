import json

class Config:
    def __init__(self, file_name):
        self.file_name = file_name

    def update(self):
        with open(self.file_name, 'r') as f:
            data = json.load(f)
            # print(data)
            for name, value in data.items():
                setattr(self, name, value)

