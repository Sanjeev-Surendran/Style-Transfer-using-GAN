FROM python:3.7.7

COPY ./docker-app /docker-app

WORKDIR /docker-app

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run app.py