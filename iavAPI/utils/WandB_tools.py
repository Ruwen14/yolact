import wandb
import json
import os
import glob


class WeightsAndBiasesAPI(object):
    """
    Implements methods as wrappers over the official API for custom stuff
    """
    def __init__(self):
        self.API = wandb.Api()

        os.makedirs('run_cache/',exist_ok=True)
    def read_run(self,run_id:str):
        """
        read data from wandb with given run_id and save it
        data is cached under .run_cache/ to speed up subsequent calls

        Args:
            run_id: id of run in wandb --> id: 'name/project-name/run-id'
        """
        wildcard = glob.glob(f'run_cache/{run_id.split("/")[-1]}*')

        # if cache of run exists --> read from it
        if (len(wildcard) is not 0) and (os.path.exists(str(wildcard[0]))):
            print(f'Found run {run_id} in cache. Load Data...')
            self.history = []
            with open(wildcard[0], 'r') as data_loader:
                for d in data_loader:
                    d = d.strip()
                    if len(d)>0:
                        self.history.append(json.loads(d))

            return self.history

        # No cache to read. Request with WandB API
        else:
            print(f'Request Data of run {run_id} from WandB...')
            self.history = []
            run = self.API.run(run_id)
            self.history = run.scan_history()
            print(f'cache run to speed up subsequent calls...')
            self.cache_run(self.history, run_id)

            return self.history

    def cache_run(self, data, run_id):
        """
        caches run to speed up subsequent calls

        Args:
            data: data to cache
            run_id: id of run
        """
        with open(f'run_cache/{run_id.split("/")[-1]}.dat', 'w+') as cache:
            for row in data:
                json.dump(row, cache)
                cache.write('\n')


class Logs(object):

    def __init__(self,path='', read=True,silent=False,data=None):
        from pathlib import Path


        self.path= Path(path)
        self.data = data
        self.silent = silent
        if read and path is not '':
            self.data= self._read_logs(path)

    def filter_data(self, filter_string=None, data=None, update=True, filter_digits=False,to_float=True):
        import re

        assert not(filter_string is not None and filter_digits==True), 'You can either filter for string or for digits'
        filtered_data = None

        if data is None:
            data = self.data

        if filter_string is not None:
            wildcard = filter_string.replace("*",'.+')
            filtered_data=[]
            for datum in data:
                s = re.search(wildcard, datum)
                if s:
                    s = s.group()
                    if not(s==''):
                        filtered_data.append(s)

            print(f'Filtered Data has {len(filtered_data)} entries') if not self.silent else None

            if update:
                self._update_data(filtered_data)

        if filter_digits:
            filtered_data = []
            for datum in data:
                if to_float:
                    filtered_data.append(float(''.join(i for i in datum if (i.isdigit() or i=='.'))))
                else:
                    filtered_data.append(''.join(i for i in datum if (i.isdigit() or i == '.')))

            if update:
                self._update_data(filtered_data)


        return filtered_data


    def get_low_percent(self,low_percentage:str, data=None):
        if data is None:
            data = self.data



        low = float(low_percentage.rstrip('%')) / 100
        amount_values = round(len(self.data) * low,0)

        data = sorted(data)
        k_lowest_values = data[:int(amount_values)]

        return (sum(k_lowest_values)/len(k_lowest_values))


    def get_percentile(self,  percentile:float, data=None):
        import numpy as np

        # assert

        if data is None:
            data = self.data

        return np.percentile(np.array(data), percentile)



    def get_avg(self, list=None):
        if list is None:
            list = self.data

        avg = sum(list)/len(list)

        return avg


    def get_min(self, list=None):
        if list is None:
            list = self.data

        return (min(list))

    def get_max(self, list=None):
        if list is None:
            list = self.data

        return (max(list))

    def eval_list(self, list=None):
        if list is None:
            list = self.data

        return dict(Max=self.get_max(list), Average=self.get_avg(list), Min=self.get_min(list))

    def _read_logs(self,file):
        with open(file, 'r') as f:
            lines = [line.rstrip() for line in f]
        print(f'Found {len(lines)} entries') if not self.silent else None
        return lines