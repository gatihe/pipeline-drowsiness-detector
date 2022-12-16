from functions import *
import cv2
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse, RedirectResponse
import os
import shutil
import random
import traceback
import requests

tags_metadata = [
    {
        "name": "data input",
        "description": "Operações de entrada de dados para previsão. É possível fornecer vídeos em formato mp4 ou conjuntos de dados com valores de EAR, MAR e PERCLOS",
    },
    {
        "name": "fit",
        "description": "Operações de treinamento de modelos de aprendizado de máquina. Para realizar o fit é necessário escolher entre 'logisticRegression', 'SVC', 'SGD'.",
    },
    {
        "name": "predict",
        "description": "Operações de aplicação de modelos de aprendizado de máquina. Caso nenhum nome de arquivo de vídeo seja passado como parâmetro, o algoritmo será aplicado para a previsão dos valores de teste estabelecidos anteriormente ao separar a base de dados entre treino e teste.",
    },
    {
        "name": "pre processing",
        "description": "Operações de tratamento de vídeos e dados para aplicação nos modelos de aprendizado de máquina.",
    },
    {
        "name": "utils",
        "description": "Operações de normalização de conjuntos de dados. Aceita como algoritmos os valores 'minMax' (normalização através do algoritmo minMax) e 'standard' (normalização através do algoritmo standard)<br>Precisa ser fornecido o nome de um conjunto de dados ",
    },
]


app = FastAPI(title='Drowsiness Detection Pipeline API', description='isto é um teste', openapi_tags=tags_metadata,
              swagger_ui_parameters={"defaultModelsExpandDepth": -1})
app.add_middleware(SessionMiddleware, secret_key="drowsiness-detector-key")

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", include_in_schema=False)
async def read_root(request: Request):
    request.session['teste'] = 'testeasd'
    return templates.TemplateResponse('index.html', {'request': request})


@app.get("/dataCleaning", include_in_schema=False)
async def dataCleaning(request: Request):
    videosList = os.listdir('src/1/uploadedVideos/')
    frameDirList = os.listdir('src/2/videoFrames/')
    return templates.TemplateResponse('dataCleaning.html', {'request': request, 'videosList':videosList, 'frameDirList':frameDirList})

@app.post("/dataCleaning", include_in_schema=False)
async def dataCleaning(request: Request, videoTimeInSeconds: Optional[int] = Form(None), selectedVideo: Optional[str] = Form(None), selectedFrameDir: Optional[str]=Form(None)):
    videosList = os.listdir('src/1/uploadedVideos/')
    frameDirList = os.listdir('src/2/videoFrames/')
    if selectedFrameDir is None and selectedVideo is not None:
        frame_count, fps, duration, framesRange, frames_dir = clipVideoFile(
        selectedVideo, videoTimeInSeconds)
    elif selectedFrameDir is not None:
        frames_dir = 'src/2/videoFrames/{}/'.format(selectedFrameDir)
        rotateImages('src/2/videoFrames/{}'.format(selectedFrameDir), getNeededRotations(selectedFrameDir))      
    else:
        frameDirList = os.listdir('src/2/videoFrames/')
    return templates.TemplateResponse('dataCleaning.html', {'request': request, 'videosList':videosList,'frameDirList':frameDirList})

@app.get("/featureEngineering", include_in_schema=False)
async def featureEngineering(request: Request, videoTimeInSeconds: Optional[int] = Form(None), selectedVideo: Optional[str] = Form(None), selectedFrameDir: Optional[str]=Form(None)):
    request.session['teste'] = 'testeasd'
    framesDir = os.listdir('src/2/videoFrames/')
    landmarksSheetsDirectory = os.listdir('src/3/landmarksSheets/')
    earMarSheetsDirectory = os.listdir('src/3/earMarSheets/')
    splitSheetsDirectory = os.listdir('src/3/splitSheets/')
    return templates.TemplateResponse('featureEngineering.html', {'request': request, 'framesDir':framesDir, 'landmarksSheetsDirectory':landmarksSheetsDirectory, 'earMarSheetsDirectory':earMarSheetsDirectory, 'splitSheetsDirectory':splitSheetsDirectory})

@app.post("/featureEngineering", include_in_schema=False)
async def featureEngineering(request: Request, selectedLandmarkSheet: Optional[str] = Form(None), videoTimeInSeconds: Optional[int] = Form(None), selectedVideo: Optional[str] = Form(None), selectedFrameDir: Optional[str]=Form(None), amountOfSeconds:Optional[str]=Form(None), selectedEarMarSheet: Optional[str]=Form(None), selectedDfToScale: Optional[str]=Form(None), selectedAlgorithm: Optional[str]=Form(None)):
    request.session['teste'] = 'testeasd'
    framesDir = os.listdir('src/2/videoFrames/')
    landmarksSheetsDirectory = os.listdir('src/3/landmarksSheets/')
    earMarSheetsDirectory = os.listdir('src/3/earMarSheets/')
    splitSheetsDirectory = os.listdir('src/3/splitSheets/')
    
    if selectedFrameDir is not None:
        fps = getFps('src/1/uploadedVideos/{}'.format(selectedFrameDir))
        framesRange = getFramesData('src/2/videoFrames/{}'.format(selectedFrameDir))
        landmarksValues = getVideoLandmarksTeste('src/2/videoFrames/{}'.format(selectedFrameDir), fps, framesRange[0], framesRange[1])
        landmarksValues.to_csv('src/3/landmarksSheets/{}.csv'.format(selectedFrameDir), index = False)
    
    elif selectedLandmarkSheet is not None: 
        landmarkDf = pd.read_csv('src/3/landmarksSheets/{}'.format(selectedLandmarkSheet))
        initialFrame = getFramesDataFromDf(landmarkDf)
        id, target, fps = getSheetTargetAndFps(selectedLandmarkSheet)
        landmarkDf['ear'] = landmarkDf.apply(lambda row: getEAR(row), axis=1)
        landmarkDf['mar'] = landmarkDf.apply(lambda row: getMAR(row), axis=1)
        landmarkDf['time'] = landmarkDf.apply(lambda row: getFrameTime(row, fps, initialFrame), axis=1)
        
        landmarkDf = landmarkDf[['id_target', 'target','ear', 'mar', 'time']]
        
        x = 0
        second = []
        meanEar = []
        meanMar = []
        allDataSeconds = pd.DataFrame()
        while x < landmarkDf['time'].max():
            second.append(x)
            meanEar.append(getSecondMeanEAR(landmarkDf, x))
            meanMar.append(getSecondMeanMAR(landmarkDf, x))
            x = x +1
            
        secondsNewSubDf = pd.DataFrame()
        secondsNewSubDf['time'] = second
        secondsNewSubDf['ear'] = meanEar
        secondsNewSubDf['mar'] = meanMar
        secondsNewSubDf['target'] = target
        secondsNewSubDf['id'] = id
        
        allDataSeconds = allDataSeconds.append(secondsNewSubDf, ignore_index = True)
        
        #transforming into 0s_ear, 1s_ear,...,
        allRows = []
        dfColumns = getColumns(secondsNewSubDf, landmarkDf['time'].max())
        values = []
        values.append('{}_{}'.format(id, target))
        for item in list(secondsNewSubDf['ear'].values):
            values.append(item)
        for item in list(secondsNewSubDf['mar'].values):
            values.append(item)
            
        values.append(target)
        allRows.append(values)
        
        earMarPerSecondDf = pd.DataFrame(allRows, columns = dfColumns)
        earMarPerSecondDf.to_csv('src/3/earMarSheets/{}'.format(selectedLandmarkSheet), index = False)
        
    elif selectedEarMarSheet is not None:  ##split video option
        id, target, fps = getSheetTargetAndFps(selectedEarMarSheet)
        earmarDf = pd.read_csv('src/3/earMarSheets/{}'.format(selectedEarMarSheet))
        videoLengthInSeconds = getAmountOfEarMarDfSeconds(earmarDf)
        earmarDf = splitDataframe(int(amountOfSeconds), earmarDf, videoLengthInSeconds)
        earmarDf.to_csv('src/3/splitSheets/{}'.format(selectedEarMarSheet), index = False)  
        
    elif selectedDfToScale is not None:
        
        dataframe = pd.read_csv('src/3/splitSheets/{}'.format(selectedDfToScale))
        dataframe = scaleDataframe(selectedAlgorithm, dataframe)
        dataframe.to_csv('src/3/scaledSheets/{}'.format(selectedDfToScale), index = False)  
        
    framesDir = os.listdir('src/2/videoFrames/')
    landmarksSheetsDirectory = os.listdir('src/3/landmarksSheets/')
    earMarSheetsDirectory = os.listdir('src/3/earMarSheets/')
    splitSheetsDirectory = os.listdir('src/3/splitSheets/')
    

    return templates.TemplateResponse('featureEngineering.html', {'request': request, 'framesDir':framesDir, 'landmarksSheetsDirectory': landmarksSheetsDirectory, 'earMarSheetsDirectory':earMarSheetsDirectory, 'splitSheetsDirectory': splitSheetsDirectory})



@app.get("/dataPreprocessing", include_in_schema=False)
async def dataPreprocessing(request: Request):
    request.session['teste'] = 'testeasd'
    return templates.TemplateResponse('dataPreprocessing.html', {'request': request})


@app.get("/modelFit", include_in_schema=False)
async def modelFit(request: Request, selectedDataset: Optional[str] = Form(None), selectedEntry: Optional[str] = Form(None), trainTestRatio: Optional[float] = Form(None), selectedFitAlgorithm: Optional[str] = Form(None)):
    availableDatasets = os.listdir('src/4/modelsDatasets/')
    
    newModelInfo = {
         "filename": "teste",
         "modelSrc": "teste",
         "dataset": "teste",
         "datasetSrc": "teste",
         "scalingAlgorithm": "teste",
         "fitAlgorithm": "teste",
         "fitDatasetSize": 0.55,
         "videosDuration": 10
    }
    
    availableModels = getSavedModelsInfo()
    availableEntries = os.listdir('src/3/scaledSheets/')
    availableDatasets = os.listdir('src/4/modelsDatasets/')
    
    
    
    return templates.TemplateResponse('modelFit.html', {'request': request, 'availableDatasets':availableDatasets, 'availableModels':availableModels, 'availableEntries': availableEntries})

@app.post("/modelFit", include_in_schema=False)
async def modelFit(request: Request, selectedDataset: Optional[str] = Form(None), selectedFitAlgorithm: Optional[str] = Form(None), trainTestRatio: Optional[str] = Form(None), usedScaleAlgorithm: Optional[str] = Form(None), clipsDuration: Optional[int] = Form(None), selectedEntry: Optional[list] = Form(None), selectedPredModel: Optional[str] = Form(None), selectedEntryPred: Optional[str] = Form(None)):
    
    results = []
    availableModels = getSavedModelsInfo()
    
    availableEntries = os.listdir('src/3/scaledSheets/')
    availableDatasets = os.listdir('src/4/modelsDatasets/') 

    if selectedDataset is not None:
        dataframe = pd.read_csv('src/4/modelsDatasets/{}'.format(selectedDataset))  
        # available entries
        
        if selectedEntry is not None:
            dataframe = dataframe.drop(columns=['perclos'])
            for item in selectedEntry:
                dfToappend = pd.read_csv('src/3/scaledSheets/{}'.format(item))    
                dataframe = pd.concat([dataframe, dfToappend], ignore_index=True)
        availableModels = getSavedModelsInfo()
        allKeys = []
        if len(availableModels.keys()) == 0:
            newModelKey = 1
        else:
            for key in availableModels.keys():
                allKeys.append(int(key))
            lastKey = max(allKeys)
            newModelKey = lastKey + 1

        X_train, X_test, y_train, y_test = trainTestSplit(dataframe, round(float(trainTestRatio), 2), selectedDataset)
        model = fitData(selectedFitAlgorithm, X_train, y_train, selectedDataset[:-4])
        testDfsPath = 'src/4/testDatasets/'
        usedInFit = 'src/4/usedInFitDf/'
        
        X_test.to_csv('{}x_test-{}_{}_{}'.format(testDfsPath,newModelKey,selectedFitAlgorithm, selectedDataset), index = False)
        y_test.to_csv('{}y_test-{}_{}_{}'.format(testDfsPath,newModelKey,selectedFitAlgorithm, selectedDataset), index = False)
        dataframe.to_csv('{}{}.csv'.format(usedInFit, newModelKey), index = False)
        
        
        
        
        newModelInfo = {
            "dataset": '{}'.format(selectedDataset),
            "scalingAlgorithm": usedScaleAlgorithm,
            "fitAlgorithm": selectedFitAlgorithm,
            "fitDatasetSize": trainTestRatio,
            "videosDuration": clipsDuration
        }
        
        addModelInfo(newModelInfo, model, selectedEntry)
        availableModels = getSavedModelsInfo()
        availableEntries = os.listdir('src/3/scaledSheets/')
        availableDatasets = os.listdir('src/4/modelsDatasets/')
    elif selectedPredModel is not None:
        dataframe = pd.read_csv('src/3/scaledSheets/{}'.format(selectedEntryPred))
        model = loadModel(selectedPredModel)
        targets_ids = list(dataframe['id_target'])
        feature_cols = list(dataframe.columns[1:-1])
        X = dataframe[feature_cols]  # Features
        y = dataframe['target']  # Target variable
        y_pred = model.predict(X)
        predResults = {'0':'0 - alerta', '5':'5 - baixa vigilância', '10':'10 - sonolento(a)'}
        results = []
        for index, item in enumerate(y_pred):
            results.append([targets_ids[index], predResults[str(item)]])
    
    return templates.TemplateResponse('modelFit.html', {'request': request, 'availableDatasets':availableDatasets, 'availableModels':availableModels, 'availableEntries':availableEntries, 'results':results})




@app.get("/fit/{selectedFitAlgorithm}/{selectedDataset}/{trainTestRatio}", tags=["fit"])
async def fit(selectedFitAlgorithm, selectedDataset, trainTestRatio, request: Request):
    try:
        dataframe = pd.read_csv('src/4/modelsDatasets/{}'.format(selectedDataset))
        X_train, X_test, y_train, y_test = trainTestSplit(dataframe, round(float(trainTestRatio), 2))
        fitData(selectedFitAlgorithm, X_train, y_train)
    except:
        raise HTTPException(
            status_code=404, detail="Invalid scaled dataframe or train size value.")
        
    availableModels = getSavedModelsInfo()
    availableEntries = os.listdir('src/3/scaledSheets/')
    availableDatasets = os.listdir('src/4/modelsDatasets/')
    return {'message':'success'}




@app.get("/prediction/{modelName}/{id}", include_in_schema=False)
async def prediction(modelName, id, request: Request):
    testDfsPath = 'src/4/testDatasets/'
    X_test = pd.read_csv('src/4/testDatasets/x_test-{}.csv'.format(modelName))
    y_test = pd.read_csv('src/4/testDatasets/y_test-{}.csv'.format(modelName))
    model = loadModel(id)
    result = predict(model, X_test, y_test)
    return templates.TemplateResponse('predict.html', {'request': request, 'result':result, 'modelName':modelName})





@app.post("/prediction/{modelName}", tags=["predict"])
async def prediction(modelName, request: Request):
    X_test = pd.read_csv('src/currentTrainTestDataset/X_test.csv')
    y_test = pd.read_csv('src/currentTrainTestDataset/y_test.csv')
    model = loadModel(modelName)
    result = predict(model, X_test, y_test)
    return result


@app.get("/predictNewEntry/{modelName}/{id}/{selectedEntry}", tags=["predict"])
async def predictVideo(modelName, id, selectedEntry, request: Request):
    X_test = pd.read_csv('src/currentTrainTestDataset/X_test.csv')
    y_test = pd.read_csv('src/currentTrainTestDataset/y_test.csv')
    model = loadModel(modelName)
    result = predict(model, X_test, y_test)
    return result



@app.get("/preprocessing/extractLandmarks/{videoFileName}/{amountOfSeconds}", tags=["pre processing"])
async def extractLandmarks(videoFileName, amountOfSeconds, request: Request):
    videoFileName = 'Ana Julia.mp4'
    frame_count, fps, duration, framesRange, frames_dir = clipVideoFile(
        videoFileName, amountOfSeconds)
    landmarksValues = getVideoLandmarks(frames_dir, fps, framesRange[0], framesRange[1])
    landmarksValues.to_csv('src/datasets/teste.csv', index = False)
    return FileResponse('src/datasets/teste.csv', media_type='text/csv', filename='scaled.csv')


@ app.get("/preprocessing/getLandmarks/{framesDirName}", tags=["pre processing"])
async def getLandmarks(videoFileName, request: Request):
    return FileResponse('src/datasets/scaled/3sec.csv', media_type='text/csv', filename='scaled.csv')


@ app.get("/preprocessing/extractFeatures/{landmarksDataframeFileName}", tags=["pre processing"])
async def extractFeatures(videoFileName, request: Request):
    return FileResponse('src/datasets/scaled/3sec.csv', media_type='text/csv', filename='scaled.csv')


@ app.get("/utils/scale/{algorithm}/{dataframeFileName}", tags=["utils"])
async def scaleData(algorithm, dataframeFileName, request: Request):
    dataframe = pd.read_csv(
        'src/3/splitSheets/{}'.format(dataframeFileName))
    dataframe = scaleDataframe(algorithm, dataframe)
    dataframe.to_csv(
        'src/3/splitSheets/{}'.format(dataframeFileName), index=False)
    return FileResponse('src/datasets/scaled/{}.csv'.format(dataframeFileName), media_type='text/csv', filename='scaled.csv')


@ app.get("/utils/split/{amountOfSeconds}/{dataframeFileName}/", tags=["utils"])
async def splitData(amountOfSeconds, dataframeFileName, request: Request):
    try:
        dataFrame = pd.read_csv('src/3/earMarSheets/{}'.format(dataframeFileName))
        videoLengthInSeconds = getAmountOfEarMarDfSeconds(dataFrame)
        splitDf = splitDataframe(int(amountOfSeconds), dataFrame, videoLengthInSeconds)
    except Exception:
        traceback.print_exc()
        return {'error': 'error'}
    return {'ok': 'ok'}

@app.post("/uploadVideo/", tags=["data input"])
async def uploadVideo(request: Request, file: UploadFile = File(...), target: str = Form()):
    with open(("src/1/uploadedVideos/{}_{}".format(target, file.filename)).replace(' ','_'), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    fps = getFps(("src/1/uploadedVideos/{}_{}".format(target, file.filename)).replace(' ','_'))
    os.rename(("src/1/uploadedVideos/{}_{}".format(target, file.filename)).replace(' ','_'), ("src/1/uploadedVideos/target{}_fps{}_{}".format(target, str(fps).replace('.',''), file.filename)).replace(' ','_'))
    return RedirectResponse(request.url_for("dataCleaning"))


@app.get("/downloadFitDataset/{id}", tags=["utils"])
async def downloadFitDataset(id, request: Request):
    usedInFit = 'src/4/usedInFitDf/'
    return FileResponse('{}{}.csv'.format(usedInFit, id), media_type='text/csv', filename='fitDatabase_{}.csv'.format(id))


@app.get("/exportModelsInfo/")
async def exportModelsInfo(request: Request):
    return FileResponse('src/4/modelsInfo/modelsInfo.json', media_type='text/json', filename='modelsInfo.json')

@app.get("/downloadModel/{id}/", tags=["utils"])
async def downloadModel(id, request: Request):
    modelDir = 'src/4/trainedModels/'
    return FileResponse('{}{}.sav'.format(modelDir, id), media_type='text/csv', filename='trainedModel_{}.sav'.format(id))