import os
import pandas as pd
import yaml

def get_example_data_file_path(filename, data_dir='example_data'):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # If you need to go up another directory (for example if you have
    # this function in your tests directory and your data is in the
    # package directory one level up) you can use
    # up_dir = os.path.split(start_dir)[0]
    #data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

class SyntheticData:
    def __init__(self, param_file, data_file, data_dir):
        data_path = get_example_data_file_path(data_file, data_dir)
        param_path = get_example_data_file_path(param_file, data_dir)
        self.params = yaml.load(open(param_path), Loader=yaml.FullLoader)
        self.data = pd.read_csv(data_path, sep='\s+', comment='#')