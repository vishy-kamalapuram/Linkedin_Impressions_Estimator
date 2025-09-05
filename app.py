from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
import io
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
    global model

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
        if not data:
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
            'prediction': int(round(float(prediction), 0)),
            'inputs': {
                'reactions': reactions,
                'comments': comments, 
                'reposts': reposts
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed:  {str(e)}'}), 500
        
            
@app.route('/predictBatch', methods=['POST'])
def predictBatch():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not (file.filename.endswith('.xlsx')):
            return jsonify({'error': 'File must be in an excel (.xlsx) format'}), 400
        
        # Ensure model is loaded correctly 
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
            
        # Read the uploaded file 
        try:
            if file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'error':f'Failed to read file : {str(e)}'}), 400
        
        # Validate required columns
        required_columns = ['REACTIONS', 'COMMENTS', 'REPOSTS']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_cols)}', 
                'required_columns': required_columns,
                'found_columns': list(df.columns)
            }), 400
        
        # Validate data types and values
        for col in required_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                return jsonify({'error': f'Column {col} must contain numeric values'}), 400
            
        # Fill missing values with zero
        # not the greatest method, but id rather predictions be made for the entire set, and almost certainly predicted
        # lower than actual, than not have any prediction
        df[required_columns] = df[required_columns].fillna(0)

        # Check for negative values
        negative_rows = df[(df[required_columns] < 0).any(axis=1)]
        if not negative_rows.empty:
            return jsonify({
                'error': f'Found negative values in rows: {negative_rows.index.tolist()}. All values must be non-negative.'
            }), 400


        input_data = df[required_columns].values
        predictions = model.predict(input_data)

        df['PREDICTED_VALUES'] = [int(round(float(pred), 0)) for pred in predictions]

        # create the output file 
        output = io.BytesIO()

        if file.filename.endswith('.xlsx'):
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')

        output.seek(0)


        base_name = file.filename.rsplit('.', 1)[0]
        output_filename = f"{base_name}_with_predictions.xlsx"
        
        return send_file(
            output,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


            
        


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)