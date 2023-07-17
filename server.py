from flask import Flask,jsonify,request
import util
app=Flask(__name__)

@app.route('/get_locations')

def get_locations():
    response = jsonify({
        'locations':util.get_locations()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price' , methods=['GET','POST'])
def predict_home_price():
    total_sqft=float(request.form['total_sqft'])
    location=request.form['location']
    bhk=int(request.form['bhk'])
    bath= int(request.form['bath'])

    response=jsonify({
        'estimated_price' : util.get_estimated_price(location,bhk,total_sqft,bath)
    })

    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route('/hello')

def hello():
    return "hiiiiiii"

if __name__=="__main__":
    print("Starting The Server ")
    util.load_saved_artifacts()
    app.run()