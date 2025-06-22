import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask,render_template,request

app=Flask(__name__)
model=joblib.load(MODEL_OUTPUT_PATH)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':

        lead_time=int(request.form['lead_time'])
        no_of_special_requests=int(request.form['no_of_special_requests'])
        avg_price_per_room=float(request.form['avg_price_per_room'])
        market_segment_type=int(request.form['market_segment_type'])
        arrival_date=int(request.form['arrival_date'])
        arrival_month=int(request.form['arrival_month'])
        no_of_week_nights=int(request.form['no_of_week_nights'])
        no_of_weekend_nights=int(request.form['no_of_weekend_nights'])
        type_of_meal_plan=int(request.form['type_of_meal_plan'])
        no_of_adults=int(request.form['no_of_adults'])

        features=np.array([[
                        lead_time,
                        no_of_special_requests,
                        avg_price_per_room,
                        market_segment_type,
                        arrival_date,
                        arrival_month,
                        no_of_week_nights,
                        no_of_weekend_nights,
                        type_of_meal_plan,
                        no_of_adults
                    ]])
        prediction=model.predict(features)

        return render_template('index.html',prediction=prediction[0])
    return render_template('index.html',prediction=None)
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)

