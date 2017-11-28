from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
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

        # Labelling data
        from sklearn import preprocessing
        from sklearn.preprocessing import OneHotEncoder

        le = preprocessing.LabelEncoder()

        workclass_cat = le.fit_transform(df.workclass)
        marital_cat   = le.fit_transform(df.marital_status)
        occupation_cat = le.fit_transform(df.occupation)
        relationship_cat = le.fit_transform(df.relationship)
        native_country_cat = le.fit_transform(df.native_country)
        education_cat = le.fit_transform(df.education)

        # Override every non-number data with its label
        df.marital_status = marital_cat
        df.occupation = occupation_cat
        df.relationship = relationship_cat
        df.native_country = native_country_cat
        df.workclass = workclass_cat

        # drop income attribute for features-extracting purpose
        # del df['income']

        # # drop less-related attribute
        del df['age']
        del df['education']
        del df['fnlwgt']

        df.loc[1] = [0, 0, 0, 0, 0, 'White', 'Male', 0, 0, 0, 0]
        df.loc[2] = [0, 0, 0, 0, 0, 'Asian-Pac-Islander', 'Female', 0, 0, 0, 0]
        df.loc[3] = [0, 0, 0, 0, 0, 'Asian-Pac-Islander', 'Female', 0, 0, 0, 0]
        df.loc[4] = [0, 0, 0, 0, 0, 'Amer-Indian-Eskimo', 'Female', 0, 0, 0, 0]
        df.loc[5] = [0, 0, 0, 0, 0, 'Black', 'Female', 0, 0, 0, 0]
        df.loc[6] = [0, 0, 0, 0, 0, 'Other', 'Female', 0, 0, 0, 0]

        encoded = pd.get_dummies(df, prefix=['race', 'sex'])
        print('PRINT encoded')
        print(encoded)

        from sklearn.externals import joblib
        model = joblib.load('best_model.pkl')
        #import numpy as np
        #example = np.array(features)
        #example = example.reshape(1, -1)

        result = model.predict(encoded.values[:,:len(encoded.loc[0])-1])

        return 'Less than or equals to 50k' if result[0] == 0 else 'More than 50k'
    else:
        return '<h1>Failed to run the process due to invalid request.<h1>'

if __name__ == '__main__':
   app.run(debug = True)