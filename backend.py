from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
   return render_template('index.html')
       

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        age = int(request.form['age'])
        workclass = request.form['workclass']
        fnlwgt = int(request.form['fnlwgt'])
        education = request.form['education']
        education_num = int(request.form['education_num'])
        marital_status = request.form['marital_status']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']
        gender = request.form['gender']
        capital_gain = int(request.form['capital_gain'])
        capital_loss = int(request.form['capital_loss'])
        hours_per_week = int(request.form['hours_per_week'])
        native_country = request.form['native_country']

        features = [('age', [age]),
                    ('workclass', [workclass]), 
                    ('fnlwgt', [fnlwgt]), 
                    ('education', [education]),
                    ('education_num', [education_num]),
                    ('marital_status', [marital_status]),
                    ('occupation', [occupation]),
                    ('relationship', [relationship]), 
                    ('race', [race]), 
                    ('gender', [gender]),
                    ('capital_gain', [capital_gain]), 
                    ('capital_loss', [capital_loss]), 
                    ('hours_per_week', [hours_per_week]),
                    ('native_country', [native_country])]

        import pandas as pd
        df = pd.DataFrame.from_items(features)

        # drop less-related attribute
        del df['age']
        del df['education_num']
        del df['fnlwgt']
        del df['native_country']
        del df['hours_per_week']

        df.loc[1] = ['State-gov', 'Bachelors', 'Never-married', 'Adm-clearical', 'Not-in-family', 'white', 'male', 0, 0]
        df.loc[2] = ['Self-emp-not-inc', 'HS-grad', 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'black', 'female', 0 ,0]
        df.loc[3] = ['Private', '11th', 'Divorced', 'Handlers-cleaners', 'Wife', 'Asian-Pac-Islander', 'male',0 ,0]
        df.loc[4] = ['Federal-gov', 'Masters', 'Married-spouse-absent', 'Prof-speciality', 'Own-child', 'Amer-Indian-Eskimo', 'male',0 ,0]
        df.loc[5] = ['Local-gov', '9th', 'Separated', 'Other-service', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[6] = ['Self-emp-inc', 'Some-college', 'Married-AF-spouse', 'Sales', 'Other-relevative', 'Other', 'male',0 ,0]
        df.loc[7] = ['Without-pay', 'Assoc-acdm', 'Divorced', 'Craft-repair', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[8] = ['Never-worked', 'Assoc-voc', 'Divorced', 'Transport-moving', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[9] = ['Never-worked', '7th-8th', 'Divorced', 'Farming-fishing', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[10] = ['Never-worked', 'Doctorate', 'Divorced', 'Machine-op-inspct', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[11] = ['Never-worked', 'Prof-school', 'Divorced', 'Tech-support', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[12] = ['Never-worked', '5th-6th', 'Divorced', 'Protective-serv', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[13] = ['Never-worked', '10th', 'Divorced', 'Armed-Forces', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[14] = ['Never-worked', '1st-4th', 'Divorced', 'Priv-house-serv', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[15] = ['Never-worked', 'Preschool', 'Divorced', 'Priv-house-serv', 'Unmarried', 'Other', 'male',0 ,0]
        df.loc[16] = ['Never-worked', '12th', 'Divorced', 'Priv-house-serv', 'Unmarried', 'Other', 'male',0 ,0] 
        df.loc[16] = ['?', '12th', 'Divorced', '?', 'Unmarried', 'Other', 'male',0 ,0] 

        col_enc = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex']
        encoded = pd.get_dummies(df, prefix=col_enc)

        from sklearn.externals import joblib
        model = joblib.load('best_model.pkl')

        result = model.predict(encoded.values[:,:len(encoded.loc[0])-1])

        result= 'Less than or equals to 50k' if result[0] == 0 else 'More than 50k'
        hasil_ = result
    else:
        hasil_ = 'Failed to run the process due to invalid request.'
    
    return jsonify(hasil = hasil_)

if __name__ == '__main__':
   app.run(debug = True)