from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


app = Flask(__name__)

# Global variable for model
model = None


def load_model():
    """
    Loads the pretrained version of the model

    Returns an integer for error codes
        0: Success
        1: Model file was not found
        2: Failed to load model (invalid model or corrupted)
    """
    """If the user does not provide a model, we will just load the existing model"""

    # model file could not be found (should never happen?)
    if not Path('models/xgbmodel.pkl').exists():
        return 1
    else:
        try:
            # Model can be successfully found and loaded
            model = joblib.load('models/xgbmodel.pkl')
            return 0
        except Exception as e:
            # Model was found, but could not be loaded
            return 2
        
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictSingle', methods=['POST'])
def predictSingle():
    try:
        data = request.get_json()

        # if data is null, return error
        if not data():
            return jsonify({'error': 'No Data provided'})
        
        # Check for missing field values
        if 'reactions' not in data:
            return jsonify({'error': 'Missing reactions value'})
        if 'comments' not in data:
            return jsonify({'error': 'Missing comments value'})
        if 'reposts' not in data:
            return jsonify({'error': 'Missing reposts value'})

        # pull the data from the response, and cast to integers
        # casting to integers because for the values, it makes sense to use integers as opposed to floats
        reactions = int(data.get('reactions'))
        comments = int(data.get('comments'))
        reposts = int(data.get('reposts'))

        # validate inputs
        # we aren't really going to hard validate here, we are just checking that negatives dont exist in the input
        # if someone were to put like 500000 reactions though, we aren't checking for that yet. 
        if reactions < 0 or comments < 0 or reposts < 0:
            return jsonify({'error': 'Invalid inputs. Values must be non-negative'})
        
        # ensure the model is loaded correctly
        if model is None:
            error_code = load_model()
            if error_code == 1:
                # again, this shouldnt happen because we already have a pretrained model 
                return jsonify({'error': 'Model file could not be found'})
            elif error_code == 2:
                # I think this shouldn't happen, but maybe if a user provides a custom set of input
                return jsonify({'error': 'Model is corrupted or invalid'})
            elif error_code != 0:
                # catch all in case there is something else that happens 
                return jsonify({'error': 'Model could not be loaded'})
            
        # Make prediction
        input_data = np.array([[reactions, comments, reposts]])
        prediction = model.predict(input_data)[0]

        # return prediction
        return jsonify({
            'prediction': round(prediction, 0),
            'inputs': {
                'reactions': reactions,
                'comments': comments, 
                'reposts': reposts
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed:  {str(e)}'}), 500
        
            


