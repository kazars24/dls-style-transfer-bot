FROM python:3.9
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir utils
RUN mkdir images
RUN mkdir images/results
RUN mkdir images/source
COPY utils/loss_and_loaders.py utils/loss_and_loaders.py
COPY utils/model.py utils/model.py
COPY tg_bot.py tg_bot.py
ENTRYPOINT ["python", "tg_bot.py"]