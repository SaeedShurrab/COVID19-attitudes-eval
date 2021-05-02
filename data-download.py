import os
import kaggle
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi


data_dir = 'data'
sub_dirs = ['raw','intermediate','preprocessed']



# data directories tree establishment 
try:
    os.mkdir(data_dir)
except:
    print(f'{data_dir} directory exists!! \n')


for dir in sub_dirs:
    try:
        os.mkdir(os.path.join(os.curdir,data_dir,dir))
    except:
        print(f'{dir} directory exists!! \n')



api = KaggleApi()
api.authenticate()


api.dataset_download_files('yazanshannak/us-covid-tweets')

print(f'download completed \n')

with ZipFile('us-covid-tweets.zip', 'r') as arch:
    arch.extractall(os.path.join('.','data','raw'))


os.remove('us-covid-tweets.zip')

print('Raw data at your service  \n')