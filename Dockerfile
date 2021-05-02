FROM python:3.8.3


RUN pip install -U pip && \
   mkdir /src \
   mkdir /root/.kaggle
   
COPY kaggle.json /root/.kaggle

WORKDIR /src


RUN pip install --no-cache-dir pandas \ 
	numpy \ 
	jupyter \
	matplotlib \ 
	nltk \
	-U scikit-learn \
	pyLDAvis \
	langdetect \
	kaggle
	
EXPOSE 8888

ENTRYPOINT jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
