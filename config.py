import configparser

def getConfig(config_name):
    configParser = configparser.RawConfigParser()   
    configFilePath = config_name
    configParser.read(configFilePath)
    return configParser

config = getConfig("config.ini") 

##### Stream config
streamCfg = config['STREAM']

##### Face detectection config
faceModel = config['FACE_MODEL']

#### Attribute analysis model
attributeModel = config['ATTRIBUTE_MODEL']

#### API config
apiConfig = config['APP_SERVER']

#### App mode
appMode = config['APP_MODE']