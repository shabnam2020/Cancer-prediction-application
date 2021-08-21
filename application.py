from flask import Flask, render_template, request, redirect, url_for
# from sklearn.externals import joblib
import joblib
import os
import pandas as pd
import json

application = Flask(__name__, template_folder='templates')
chosen_model = joblib.load("./models/chosen_model.pkl")
encoder = joblib.load("./models/label_encoder.pkl")
credentials = "credentials.csv"
feedbacks = "feedbacks.csv"
is_logged_in = False
logged_in_user = ""

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# WELCOME PAGE
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


@application.route("/")
@application.route("/welcome")
# application = Flask(__name__, )
def welcome():
    # feeds = pd.read_csv(feedbacks, header=0)
    # print(feeds.values)
    # stocklist = list(feeds.values)
    # print(stocklist)

    return render_template("welcome.html")
    # return render_template("index.html", stocklist=stocklist)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# REGISTER PAGE
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


@application.route("/login", methods=['GET', 'POST'])
def register():
    try:
        msg = eval(request.args['messages'])
    except:
        msg = {"msg": ""}
    return render_template("login.html", data=msg["msg"])


@ application.route("/registered", methods=['POST'])
def registered():
    if request.method == 'POST':
        if request.form.get('RegisterBtn') == 'RegisterBtn':
            registered_data = request.form.to_dict()
            registered_data.pop("RegisterBtn")
            if query_for_email(registered_data["email"])[0] is None:
                if registered_data["pwd"] == registered_data["pwd-repeat"]:
                    registered_info = pd.DataFrame(
                        [registered_data.values()], columns=registered_data.keys())
                    creds = pd.read_csv(credentials)
                    creds = pd.concat([creds, registered_info], axis=0)
                    creds.to_csv(credentials, index=False)
                    print(creds)
                else:
                    msg = "Passwords did not match!!"
                    print(msg)
                    return redirect(url_for('login', messages={"msg": msg}))
            else:
                return redirect(url_for('login', messages={"msg": "Already registerd user"}))
                # already registerd user check

    return render_template("login.html")

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# LOGIN PAGE
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


@ application.route("/login", methods=['GET', 'POST'])
def login():
    if is_logged_in:
        return redirect("/home")
    else:
        try:
            messages = eval(request.args['messages'])
        except:
            messages = {"msg": ""}
        return render_template("login.html", data=messages["msg"])


@ application.route("/loggedin", methods=['POST'])
def loggedin():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('LoginBtn') == 'LoginBtn':
            login_data = request.form.to_dict()
            login_data.pop("LoginBtn")
            logged_in_user = login_data['email']
            email, pwd = query_for_email(logged_in_user)
            if email is not None and pwd is not None:
                print(email)
                print(logged_in_user)
                print(pwd)
                print(login_data['pwd'])

                if email == str(logged_in_user) and pwd == str(login_data['pwd']):
                    is_logged_in = True
                    messages = {"login": is_logged_in, "user": logged_in_user}
                    return redirect(url_for('home', messages=messages))
                else:
                    is_logged_in = False
                    logged_in_user = ""
                    return redirect(url_for('login', messages={"msg": "Wrong Email or Password!!"}))
                    print(is_logged_in)
    return redirect("/login")

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# FEEDBACK PAGE
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


@ application.route("/feedback", methods=['GET', 'POST'])
def feedback():
    feeds = pd.read_csv(feedbacks, header=0)
    print(feeds.values)
    stocklist = list(feeds.values)
    print(stocklist)
    return render_template("feedback.html", feeds=stocklist)


@ application.route("/send_feedback", methods=['POST'])
def send_feedback():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('SendFeedback') == 'SendFeedback':
            feedback_data = request.form.to_dict()
            feedback_data.pop("SendFeedback")
            print(feedback_data)

            feedback_info = pd.DataFrame(
                [feedback_data.values()], columns=feedback_data.keys())
            feeds = pd.read_csv(feedbacks)
            feeds = pd.concat([feeds, feedback_info], axis=0)
            feeds.to_csv(feedbacks, index=False)
            print(feeds)

    return redirect("/feedback")

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# HOME PAGE
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


@ application.route("/home")
def home():
    try:
        messages = eval(request.args['messages'])
        if messages["login"] == True:
            return render_template("home.html", data=messages["user"])
        else:
            return redirect("/login")
    except:
        return redirect("/login")

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# CANCER PREDICTION PAGE
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////


@ application.route("/predict", methods=['GET', 'POST'])
def predict():
    print(request.method)
    result_class = "Unable to Predict"
    if request.method == 'POST':
        if request.form.get('submit') == 'Submit':
            print("Form Submitted!!")
            # data input
            input_data = request.form.to_dict()
            input_data.pop("submit")
            df = pd.DataFrame([input_data.values()], columns=input_data.keys())
            print(df)
            # return (request.headers)
            try:
                # predict class
                pred = chosen_model.predict(df)
                # inverse transform class
                result_class = encoder.inverse_transform(pred)[0]
                print(pred, result_class)
            except Exception as e:
                print("Unable to predict!!")
                return render_template("error.html", data=e)

        elif request.form.get('Logout') == 'Logout':
            is_logged_in = False
            logged_in_user = ""
            return redirect("/login")
        else:
            print("Form NOT Submitted.")
            return redirect(url_for('home'))
    elif request.method == 'GET':
        print("No Post Back Call")
        return redirect('/home')
    return redirect(url_for('predicted', messages={"result_class": result_class}))


@ application.route("/predicted")
def predicted():
    try:
        msg = eval(request.args['messages'])
    except:
        msg = {"result_class": ""}

    if msg["result_class"] == "B":
        returnMsg = "Benign"
    elif msg["result_class"] == "M":
        returnMsg = "Malignant"

    return render_template("predict.html", data=[msg["result_class"], returnMsg])


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# HELPER FUNCTIONS
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
def query_for_email(email):
    creds = pd.read_csv(credentials)
    query = creds.query(f"email == '{email}'")
    if len(query) >= 1:
        pwd = str(query["pwd"].values[0])
        email = str(query["email"].values[0])
        return email, pwd
    else:
        return None, None


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN FUNCTION
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    port = 8088
    # print('application.port ' + port)
    # port = int(os.environ.get('PORT', 8088))
    application.run(host='0.0.0.0', port=port, debug=True)
