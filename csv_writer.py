import csv

class CsvWriter:
    def __init__(self, directory, headers):
        self._log_directory = directory
        self._headers = headers
        with open(directory, 'w', newline='') as data_file:
            writer = csv.DictWriter(data_file, headers)
            writer.writeheader()

    def write_data(self, data):
        with open(self._log_directory, mode='a', newline='') as data_file:
            writer = csv.DictWriter(data_file, self._headers)
            writer.writerow(dict(zip(self._headers, data)))