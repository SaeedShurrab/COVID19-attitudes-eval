import os
import zipfile

data_dir = 'data'
sub_dirs = ['raw','intermediate','preprocessed']



# data directories tree establishment 
try:
    os.mkdir(data_dir)
except:
    print(f'{data_dir} directory exists!!')


for dir in sub_dirs:
    try:
        os.mkdir(os.path.join(os.curdir,data_dir,dir))
    except:
        print(f'{dir} directory exists!!')





#UN data download
#UN_URL = 'https://justdata91.s3.us-east-2.amazonaws.com/UNv1.0.ar-en.ar.tar.xz'
#response = requests.get(UN_URL)
#UN_dir = os.path.join(raw_dir,data_sources[0])

#if response.status_code == 200:
#    with open(os.path.join(UN_dir,'UNv1.0.ar-en.ar.tar.xz'), "wb+") as file:
#        file.write(response.content)
#        print("Download completed")
#else:
#    print("Download Failed!!")

#with tarfile.open(os.path.join(UN_dir,'UNv1.0.ar-en.ar.tar.xz'),'r') as archive:
#    archive.extractall(UN_dir)
#    os.remove(os.path.join(UN_dir,'UNv1.0.ar-en.ar.tar.xz'))