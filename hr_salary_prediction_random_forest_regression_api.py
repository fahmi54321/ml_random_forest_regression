from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)


with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


def get_salary_category(salary):
    if salary < 100000:
        return "Low to Mid Level"
    elif salary < 300000:
        return "Mid to Senior Level"
    else:
        return "Executive Level"

def get_confidence_note(level):
    if level < 3 or level > 9:
        return "Prediction may be less accurate due to edge data."
    return "Prediction is within normal range."

def get_recommendation(prediction):
    if prediction < 100000:
        return "Consider negotiating or improving skills."
    elif prediction < 300000:
        return "Competitive salary range."
    else:
        return "High-level salary, strong negotiation position."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        level = float(data['position_level'])


        prediction = model.predict([[level]])[0]


        x_grid = np.arange(1, 10, 0.1).reshape(-1, 1)
        y_pred = model.predict(x_grid)

        curve = [
            {"x": float(x), "y": float(y)}
            for x, y in zip(x_grid.flatten(), y_pred.flatten())
        ]


        real_data = [
            {"x": 1, "y": 45000},
            {"x": 2, "y": 50000},
            {"x": 3, "y": 60000},
            {"x": 4, "y": 80000},
            {"x": 5, "y": 110000},
            {"x": 6, "y": 150000},
            {"x": 7, "y": 200000},
            {"x": 8, "y": 300000},
            {"x": 9, "y": 500000},
            {"x": 10, "y": 1000000},
        ]

        user_point = {
            "x": level,
            "y": float(prediction)
        }


        return jsonify({
            "input": {
                "position_level": level
            },
            "prediction": {
                "salary": round(float(prediction), 2),
                "currency": "USD",
                "formatted": f"${int(prediction):,}"
            },
            "insight": {
                "category": get_salary_category(prediction),
                "confidence_note": get_confidence_note(level),
                "recommendation": get_recommendation(prediction)
            },
            "meta": {
                "model": "Random Forest Regression",
                "note": "No feature scaling applied"
            },
            "visualization": {
                "real_data": real_data,
                "curve": curve,
                "user_point": user_point
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)