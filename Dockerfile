FROM python:3.7
RUN apt update -y
RUN apt install python3-pip python3-dev build-essential -y
COPY . /fakeapp
WORKDIR /fakeapp
ENV PORT $PORT
RUN pip install -r requirements.txt
RUN gdown https://drive.google.com/uc?id=1f_QqiENi3rliBri48J5MTcsYBae0T0c7
ENTRYPOINT ["python"]
CMD ["app.py"]

