from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model artifacts
with open('encoder', 'rb') as f:
    oh = pickle.load(f)

with open('scaler', 'rb') as f:
    sc = pickle.load(f)

with open('model', 'rb') as f:
    reg = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        rd = float(data['rd_spend'])
        ad = float(data['administration'])
        mar = float(data['marketing_spend'])
        state = data['state']

        d1 = {
            'R&D Spend': [rd],
            'Administration': [ad],
            'Marketing Spend': [mar],
            'State': [state]
        }

        newdf = pd.DataFrame(d1)
        newdf[oh.get_feature_names_out()] = oh.transform(newdf[['State']])
        newdf.drop(columns=oh.feature_names_in_, inplace=True)
        newdf[['R&D Spend', 'Administration', 'Marketing Spend']] = sc.transform(
            newdf[['R&D Spend', 'Administration', 'Marketing Spend']]
        )

        result = reg.predict(newdf)
        final = float(result[0])

        return jsonify({'success': True, 'profit': final})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)