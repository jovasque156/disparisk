import os
import logging
import pandas as pd
import numpy as np
import twitteraae.code
from twitteraae.code import predict

# Setting logger
extra = {'app_name':__name__}

# Gets or creates a logger
logger = logging.getLogger(__name__)

# stream handler and fomatter
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(app_name)s : %(message)s')
syslog.setFormatter(formatter)

# set logger level and add handlers
logger.setLevel(logging.INFO)
logger.addHandler(syslog)

# set logger adapter
logger = logging.LoggerAdapter(logger, extra)

_HATE_SPEECH_PATH = '/path/to/data_sets/hate_speech/'

if __name__ == '__main__':
        logger.info('Starting processing')
        
        logger.info('Loading dataset')
        ds = pd.read_csv(os.path.join(_HATE_SPEECH_PATH, 'labeled_data.csv'))
        ds = ds.rename(columns={'Unnamed: 0': 'id'})
        logger.info('Dataset loaded')

        logger.info('Making AAE predictions')
        predict.load_model()

        ds['predictions'] = ds['tweet'].apply(lambda tweet: predict.predict(tweet.split()))

        # Extract predictions into separate lists
        aa = [pred[0] 
                if pred is not None else None 
                for pred in ds['predictions']]
        hispanic = [pred[1]
                if pred is not None else None 
                for pred in ds['predictions']]
        other = [pred[2]
                if pred is not None else None 
                for pred in ds['predictions']]
        white = [pred[3]
                if pred is not None else None
                for pred in ds['predictions']]
        logger.info('Predictions done. Adding to the dataset and saving it into a csv file.')

        # Optionally, you can also assign these lists back to the DataFrame if needed
        ds['aa'] = aa
        ds['hispanic'] = hispanic
        ds['other'] = other
        ds['white'] = white
        
        d_classes = []
        t_classes = []

        for i in range(len(ds)):
                if ds.iloc[i,-4:].isna().any():
                        d_classes.append('NaN')
                else:
                        d_classes.append(ds.columns[-4:][ds.iloc[i,-4:].argmax()])
                t_classes.append(ds.columns[2:5][ds.iloc[i,5]])

        ds['dialect_class'] = d_classes
        ds['class_label'] = t_classes
        ds = ds[ds['dialect_class']!='NaN']

        ds.index = range(len(ds))

        ds.to_csv(os.path.join(_HATE_SPEECH_PATH, 'labeled_data_AAE.csv'))
        logger.info('csv file saved.')