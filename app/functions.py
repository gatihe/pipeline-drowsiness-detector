import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy
from scipy import spatial
from scipy.spatial import distance
from numpy.ma.core import empty
import pickle
import cv2
import os
import random
import dlib
from imutils import face_utils
import json


def getSavedModelsInfo():
    f = open('src/4/modelsInfo/modelsInfo.json')
    modelsInfo = json.load(f)
    f.close()
    return modelsInfo

def addModelInfo(newModelInfo, model, selectedEntry):
    f = open('src/4/modelsInfo/modelsInfo.json')
    modelsInfo = json.load(f)
    f.close()
    
    #getting updated key for new model
    allKeys = []
    if len(modelsInfo.keys()) == 0:
        newModelKey = 1
    else:
        for key in modelsInfo.keys():
            allKeys.append(int(key))
        lastKey = max(allKeys)
        
        newModelKey = lastKey + 1
        
        
    
    #assembling new model info
    
    modelName = '{}_{}_{}'.format(newModelKey, newModelInfo['fitAlgorithm'], newModelInfo['dataset'][:-4])
    
    datasetName = newModelInfo['dataset']
    if selectedEntry is not None:
        for item in selectedEntry:
            datasetName = datasetName + ' + {}'.format(item)
    
    newModel = {
        str(newModelKey):
            {
                "filename": modelName,
                "dataset": datasetName,
                "scalingAlgorithm": newModelInfo['scalingAlgorithm'],
                "fitAlgorithm": newModelInfo['fitAlgorithm'],
                "fitDatasetSize": newModelInfo['fitDatasetSize'],
                "videosDuration": newModelInfo['videosDuration']
            }
    }
    
    modelsInfo.update(newModel)
    
    #saving
    if os.path.exists('src/4/modelsInfo/modelsInfo.json'):
        os.remove('src/4/modelsInfo/modelsInfo.json')
        
        
    json_object = json.dumps(modelsInfo, indent=4)
    # Writing to sample.json
    with open("src/4/modelsInfo/modelsInfo.json", "w") as outfile:
        outfile.write(json_object)
        
        
    
    saveModel(model, str(newModelKey))
        
    return modelsInfo

def removeJpgFromFrameStr(row):
    return int(row['frame'].replace('.jpg', ''))

def convertIntoTimeSeries(earMarkAndLandmarksDf):
    earMarkAndLandmarksDf = earMarkAndLandmarksDf[['id_target', 'target', 'frame','mar', 'ear', 'fps']]
    earMarkAndLandmarksDf['frame_int'] = earMarkAndLandmarksDf.apply(lambda row: removeJpgFromFrameStr(row), axis=1)
    earMarkAndLandmarksDf['frame_int']
        
    return earMarkAndLandmarksDf

def splitDataframe(amountOfSeconds, dataFrame, videoLengthInSeconds):
    x = 0
    initialClip = 0
    endClip = amountOfSeconds
    splitedColumns = []
    while x <= videoLengthInSeconds:
        testDf = []
        for i in list(dataFrame.columns):
            if 'ear' in i or 'mar' in i:
                column = i.replace('s_ear', '')
                column = column.replace('s_mar', '')
                if int(column) in range(x, x+amountOfSeconds):
                    testDf.append(i)
            else:
                testDf.append(i)
                
        splitedColumns.append(testDf)
        if x == 0:
            newColumns = testDf
            splittedDf = pd.DataFrame(columns=newColumns)
        x = x + amountOfSeconds

    for item in splitedColumns:
        newDf = dataFrame[item]
        newDf.columns = newColumns
        splittedDf = splittedDf.append(newDf, ignore_index=True)

    return splittedDf

def getAmountOfEarMarDfSeconds(earmarDf):
    secondsValues = []
    for item in list(earmarDf.columns):
        if 'ear' in item:
            secondsValues.append(int(item.replace('s_ear', '')))
            
    return max(secondsValues)

def getSheetTargetAndFps(sheetname):
    fileNameParts = sheetname.split('_')
    for item in fileNameParts:
        if 'fps' in item:
            fpsPart = item
            fps = round(float(int(item.replace('fps',''))/10),2)
        if 'target' in item:
            target = item.replace('target','')
            
    fileNameParts = sheetname.split(fpsPart)
    id = fileNameParts[-1].replace('_','').replace('.csv','')
    return id, target, fps


def getSecondMeanEAR(df, second):
  df = df[df['time'] >= second]
  df = df[df['time'] < second + 1]
  return df['ear'].mean()

def getSecondMeanMAR(df, second):
  df = df[df['time'] >= second]
  df = df[df['time'] < second + 1]
  return df['mar'].mean()

def updatePerclos(row, minEyeOpenessThreshold, amountOfSeconds):
    x = 0
    eyesClosedSeconds = 0
    while x < amountOfSeconds:
        ear = row['{}s_ear'.format(x)]
        if ear <= minEyeOpenessThreshold:
            eyesClosedSeconds = eyesClosedSeconds + 1
        x = x + 1

    if eyesClosedSeconds > 0:
        return eyesClosedSeconds/amountOfSeconds
    else:
        return 0

def getColumns(videoDf, amountOfValuesForEARMAR):
      columnList = []
      x = 0
      newColumns = []
      newColumns.append('id_target')
      earSecColumns = []
      marSecColumns = []
      while x < amountOfValuesForEARMAR:
        earSecColumnID = '{}s_ear'.format(str(x))
        marSecColumnID = '{}s_mar'.format(str(x))
        earSecColumns.append(earSecColumnID)
        marSecColumns.append(marSecColumnID)
        x = x + 1
      for item in earSecColumns:
        newColumns.append(item)
      for item in marSecColumns:
        newColumns.append(item)
      newColumns.append('target')
      return newColumns


def trainTestSplit(dataframe, trainSize, selectedDataset):
    testDfsPath = 'src/4/testDatasets/'
    feature_cols = list(dataframe.columns[1:-1])
    X = dataframe[feature_cols]  # Features
    y = dataframe['target']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=trainSize, random_state=16)
    return X_train, X_test, y_train, y_test


def fitData(algorithm, X_train, y_train, datasetName):

    if algorithm == 'logisticRegression':
        model = LogisticRegression(random_state=16)
    if algorithm == 'SGD':
        model = SGDClassifier(max_iter=1, tol=0.01)
    if algorithm == 'SVC':
        model = svm.SVC(kernel='poly', gamma='scale', max_iter=1)

    model.fit(X_train, y_train)
    return model


def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


def saveModel(model, name):
    pickle.dump(model, open('src/4/trainedModels/{}.sav'.format(name), 'wb'))


def loadModel(name):
    model = pickle.load(open('src/4/trainedModels/{}.sav'.format(name), 'rb'))
    return model


def scaleDataframe(algorithm, dataframe):

    if algorithm == 'minMax':
        scaledDf = dataframe.T.iloc[1:,:]
        scaledDf = scaledDf.reset_index(drop=True)

        x = 0
        while x < len(list(scaledDf.columns)):
            scaledDf.iloc[:,x] = MinMaxScaler().fit_transform(np.array(scaledDf.iloc[:,x]).reshape(-1,1))
            x = x + 1

        scaledDf = scaledDf.T

        scaledDf.insert(0,'id', list(dataframe['id_target'].values))
        scaledDf.columns = dataframe.columns
        scaledDf['target'] = dataframe['target']
    if algorithm == 'standard':
        scaledDf = dataframe.T.iloc[1:,:]
        scaledDf = scaledDf.reset_index(drop=True)

        x = 0
        while x < len(list(scaledDf.columns)):
            scaledDf.iloc[:,x] = StandardScaler().fit_transform(np.array(scaledDf.iloc[:,x]).reshape(-1,1))
            x = x + 1
        scaledDf = scaledDf.T
        scaledDf.insert(0,'id', list(dataframe['id_target'].values))
        scaledDf.columns = dataframe.columns
        scaledDf['target'] = dataframe['target']
        scaledDf

    return scaledDf

def getMetrics(modelName, id):
    testDfsPath = 'src/4/testDatasets/'
    X_test = pd.read_csv('src/4/testDatasets/x_test-{}.csv'.format(modelName))
    y_test = pd.read_csv('src/4/testDatasets/y_test-{}.csv'.format(modelName))
    model = loadModel(id)
    result = predict(model, X_test, y_test)
    return result

def getFps(videoSource):
    cap = cv2.VideoCapture(videoSource)
    return cap.get(cv2.CAP_PROP_FPS)


def clipVideoFile(videoFileName, amountOfSeconds):
    cap = cv2.VideoCapture("src/1/uploadedVideos/{}".format(videoFileName))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    if amountOfSeconds > duration:
        amountOfSeconds = duration
    amountOfFrames = int(amountOfSeconds) * fps
    randomStartRange = random.randint(1, frame_count-int(amountOfFrames))
    framesRange = [int(randomStartRange), int(
        randomStartRange + amountOfFrames)]
    frames_dir = 'src/2/videoFrames/{}/'.format(videoFileName[:-4])
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    id_frame = 1
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frames_dir, str(id_frame) + ".jpg")
            if id_frame in range(framesRange[0], (framesRange[1]+1)):
                cv2.imwrite(frame_path, frame)
            id_frame += 1
    cap.release()
    return frame_count, fps, duration, framesRange, frames_dir

def getFramesData(frames_dir):
    frames = os.listdir(frames_dir)
    intFrames = []
    for frame in frames:
        intFrames.append(int(frame[:-4]))
    
    framesRange = []
    framesRange.append(max(intFrames))
    framesRange.append(min(intFrames))
    return framesRange

def getFramesDataFromDf(df):
    frameValues = list(df['frame'])
    newFrameValues = []
    for item in frameValues:
        newFrameValues.append(int(item.replace('.jpg', '')))
    return min(newFrameValues)    
    

def getVideoLandmarks(frames_dir, fps, startFrame, lastFrame):
    framesToRead = os.listdir(frames_dir)
    linesToAppend = []
    landmarksDf = pd.DataFrame(columns = generateLandmarkColumns())
    for item in framesToRead:
        landmarksRow = getImageLandmarks(
            frames_dir+item, generateLandmarkColumns(), 'teste', '10', item)
        linesToAppend.append(landmarksRow)
    for landmarkRow in linesToAppend:
        landmarksDf = landmarksDf.append(landmarkRow, ignore_index = True)
    return landmarksDf

def getVideoLandmarksTeste(frames_dir, fps, startFrame, lastFrame):
    framesToRead = os.listdir(frames_dir)
    linesToAppend = []
    frameOrder = []
    landmarksDf = pd.DataFrame(columns = generateLandmarkColumns())
    for frame in framesToRead:
        frameOrder.append(int(frame[:-4]))
    frameOrder.sort()
    orderedFrames = []
    for item in frameOrder:
        orderedFrames.append(str(item)+'.jpg')
    
    for item in orderedFrames:
        landmarksRow = getImageLandmarks(
            frames_dir+'/'+item, generateLandmarkColumns(), 'teste', '10', item)
        linesToAppend.append(landmarksRow)
    for landmarkRow in linesToAppend:
        landmarksDf = landmarksDf.append(landmarkRow, ignore_index = True)
        
    landmarksDf = landmarksDf.assign(Fps=fps)
    return landmarksDf


def generateLandmarkColumns():
    landmarkColumns = []
    landmarkColumns.append('id_target')
    landmarkColumns.append('target')
    landmarkColumns.append('frame')
    i = 1
    while i < 69:
        columnString = 'Landmark '+str(i)
        landmarkColumns.append(columnString)
        i = i + 1
    return landmarkColumns


def getImageLandmarks(imgPath, landmarkColumns, videoName, target, frame):
    p = "shape_predictor/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    image = cv2.imread(imgPath)
    videoLandmarks = []
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image, 0)
        if len(rects) > 0:
            for (i, rect) in enumerate(rects):
                shape = predictor(gray_image, rect)
                shape = face_utils.shape_to_np(shape)
                videoLandmarks.append(videoName)
                videoLandmarks.append(target)
                videoLandmarks.append(str(frame))
                j = 0

            for k in shape:
                videoLandmarks.append(k)
                j = j+1
        else:
            videoLandmarks.append(videoName)
            videoLandmarks.append(target)
            videoLandmarks.append(str(frame))
            print('Invalid frame {}'.format(str(frame)))
            h = 1
            while h < 69:
                emptyArray = []
                videoLandmarks.append(emptyArray)
                h = h+1

        landmarksDict = {}
        for a, b in zip(landmarkColumns, videoLandmarks):
            landmarksDict[a] = b
        return landmarksDict
    else:
        print('Couldnt open {}'.format(str(frame)+'.jpg'))
        videoLandmarks.append(videoName)
        videoLandmarks.append(target)
        videoLandmarks.append(str(frame))
        h = 1
        while h < 69:
            emptyArray = ['NA']
            videoLandmarks.append(emptyArray)
            h = h+1

        landmarksDict = {}
        for a, b in zip(landmarkColumns, videoLandmarks):
            landmarksDict[a] = b
        return landmarksDict

# calcula a quantidade de rotações necessárias
def getNeededRotations(video):
  for i in os.listdir('src/2/videoFrames/{}'.format(video)):
    framePath = 'src/2/videoFrames/{}/{}'.format(video, i)
    image = cv2.imread(framePath)
    height, width, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectando as faces em preto e branco.
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray_image, 0)
    rotationsNeeded = 0
    while(len(rects) == 0 ):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_image)
        rotationsNeeded = rotationsNeeded + 1
        if len(rects) != 0:
          break
        if rotationsNeeded == 3 and len(rects) == 0:
          print('didnt find correct position for frame: '+frame)
          break
    if len(rects) > 0:
      print('image orientation is correct')
      break
  return rotationsNeeded

def rotateImages(frameDir, rotationsNeeded):
    if rotationsNeeded > 0:
        for frame in os.listdir(frameDir):
            image = cv2.imread('{}/{}'.format(frameDir, frame))
            if image is not None:
                image = np.rot90(image,rotationsNeeded).copy() #rotacionando o frame com numpy
                cv2.imwrite('{}/{}'.format(frameDir, frame) , image)
    return None

# converte o valor obtido no dataframe salvo de string para um array numérico
def lmStringToArray(landmarkString):
  if 'NA' not in landmarkString and len(landmarkString) > 0 and str(landmarkString) != '[]':
    try:
      firstPiece = int(landmarkString[1:4])
      secondPiece = int(landmarkString[5:8])
    except ValueError:
      firstPiece = -1
      secondPiece = -1
  else:
    firstPiece = -1
    secondPiece = -1
  return [firstPiece, secondPiece]

def getLastModelId():
    availableModels = getSavedModelsInfo()
    usedKeys = []
    for key in availableModels:
        usedKeys.append(int(key))
        
    return max(usedKeys)

# calcula o valor do EAR para um olho
def getEyeAspectRatio(eye):
    p2_minus_p6 = distance.euclidean(eye[1], eye[5])
    p3_minus_p5 = distance.euclidean(eye[2], eye[4])
    p1_minus_p4 = distance.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

# converte o valor obtido no dataframe salvo de string para um array numérico
def lmStringToArray(landmarkString):
  if 'NA' not in landmarkString and len(landmarkString) > 0 and str(landmarkString) != '[]':
    try:
      firstPiece = int(landmarkString[1:4])
      secondPiece = int(landmarkString[5:8])
    except ValueError:
      firstPiece = 0
      secondPiece = 0
  else:
    firstPiece = 0
    secondPiece = 0
  return [firstPiece, secondPiece]

#calcula o tempo do frame dado um fps
def getFrameTime(row, fps, initialFrame):
    time = round((int(row['frame'].replace('.jpg', ''))-initialFrame) / fps,2)
    return time

#calcula o valor do EAR para uma coluna
def getEAR(row):

  rightEye = [
              lmStringToArray(row['Landmark 37']), #P1
              lmStringToArray(row['Landmark 38']), #P2
              lmStringToArray(row['Landmark 39']),  #P3
              lmStringToArray(row['Landmark 40']), #P4
              lmStringToArray(row['Landmark 41']), #P5
              lmStringToArray(row['Landmark 42']) #P6
              ]
      
  leftEye = [
              lmStringToArray(row['Landmark 43']), #P1
              lmStringToArray(row['Landmark 44']),  #P2
              lmStringToArray(row['Landmark 45']),  #P3
              lmStringToArray(row['Landmark 46']),  #P4
              lmStringToArray(row['Landmark 47']),  #P5
              lmStringToArray(row['Landmark 48']) #p6
              ]

  if -1 in rightEye or -1 in leftEye:
    return np.nan

  leftEAR = (distance.euclidean(leftEye[1], leftEye[5]) + distance.euclidean(leftEye[2], leftEye[4]) ) / (2.0 * distance.euclidean(leftEye[0], leftEye[3]))
  rightEAR = (distance.euclidean(rightEye[1], rightEye[5]) + distance.euclidean(rightEye[2], rightEye[4]) ) / (2.0 * distance.euclidean(rightEye[0], rightEye[3]))
  ear = (leftEAR + rightEAR)/2
  return ear

#calcula o valor do MAR para uma coluna
def getMAR(row):
  p1 = lmStringToArray(row['Landmark 61'])
  p2 = lmStringToArray(row['Landmark 62'])
  p3 = lmStringToArray(row['Landmark 63'])
  p4 = lmStringToArray(row['Landmark 64'])
  p5 = lmStringToArray(row['Landmark 65'])
  p6 = lmStringToArray(row['Landmark 66'])
  p7 = lmStringToArray(row['Landmark 67'])
  p8 = lmStringToArray(row['Landmark 68'])
  p2_minus_p8 = distance.euclidean(p2, p8)
  p3_minus_p7 = distance.euclidean(p3, p7)
  p4_minus_p6 = distance.euclidean(p4, p6)
  p1_minus_p5 = distance.euclidean(p1, p5)
  mar = ((p2_minus_p8) + (p3_minus_p7) + (p4_minus_p6))/(3*(p1_minus_p5))
  return mar
