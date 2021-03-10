import csv

class CsvWriter:

    EPOCH_HEADERS = ["Episode Number", "Maximum Reward", "Total Reward", "Exploring Rate"]
    EPISODE_HEADERS = ["Position", "Location", "Action", "Exploring", "Reward"]

    def __init__(self, directory, headers):
        self._log_directory = directory
        self._headers = headers
        with open(directory, 'w', newline='') as data_file:
            writer = csv.DictWriter(data_file, headers)
            writer.writeheader()

    def write_vector(self, data):
        with open(self._log_directory, mode='a', newline='') as data_file:
            writer = csv.DictWriter(data_file, self._headers)
            writer.writerow(dict(zip(self._headers, data)))

    def write_matrix(self, data):
          with open(self._log_directory, mode='a', newline='') as data_file:
            writer = csv.DictWriter(data_file, self._headers)
            for vector in data:
                writer.writerow(dict(zip(self._headers, vector)))