#Docker image with needed steps to quick launch application.


# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pipeline-base

COPY ./app /app
COPY requirements.txt /app

WORKDIR /app

RUN python3 -m pip install -r requirements.txt 


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "80"]
