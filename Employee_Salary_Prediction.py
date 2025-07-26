#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Employee Salary Prediction using csv
#Load your Libarary
import pandas as pd


# In[2]:


data=pd.read_csv(r"C:\Users\NITHYA SRI\Downloads\adult 3.csv")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.head(7)


# In[7]:


data.tail()


# In[8]:


data.tail(8)


# In[9]:


#Null Values
data.isna()


# In[10]:


data.isna().sum()


# In[11]:


print(data.occupation.value_counts())


# In[12]:


print(data.gender.value_counts())


# In[13]:


print(data.workclass.value_counts())


# In[14]:


print(data.education.value_counts())


# In[15]:


print(data["marital-status"].value_counts())


# In[16]:


data.occupation.replace({'?':'Others'},inplace=True)


# In[17]:


print(data.occupation.value_counts())


# In[18]:


data.workclass.replace({'?':'Others'},inplace=True)


# In[19]:


print(data.workclass.value_counts())


# In[20]:


data


# In[21]:


print(data.workclass.value_counts())


# In[22]:


data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']


# In[23]:


print(data.workclass.value_counts())


# In[24]:


data.shape


# In[25]:


print(data.education.value_counts())


# In[26]:


data=data[data['education']!='5th-6th']
data=data[data['education']!='1st-4th']
data=data[data['education']!='Preschool']


# In[27]:


print(data.education.value_counts())


# In[28]:


data.shape


# In[29]:


data


# In[30]:


#redundancy
data.drop(columns=['education'],inplace=True)


# In[31]:


#get_ipython().run_line_magic('pip', 'install matplotlib')


# In[32]:


#outlier
import matplotlib.pyplot as plt 
plt.boxplot(data['age'])
plt.show()


# In[33]:


plt.boxplot(data['educational-num'])
plt.show()


# In[34]:


plt.boxplot(data['fnlwgt'])
plt.show()


# In[35]:


plt.boxplot(data['capital-gain'])
plt.show()


# In[36]:


plt.boxplot(data['hours-per-week'])
plt.show()


# In[37]:


data=data[(data['age']<=75)&(data['age']>=17)]


# In[38]:


plt.boxplot(data['age'])
plt.show()


# In[39]:


data


# In[40]:


#get_ipython().run_line_magic('pip', 'install scikit-learn')


# In[41]:


#Label encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['capital-gain']=encoder.fit_transform(data['capital-gain'])
data['capital-loss']=encoder.fit_transform(data['capital-loss'])
data['native-country']=encoder.fit_transform(data['native-country'])
data


# In[42]:


x=data.drop(columns=['income']) #input
y = data['income']


# In[43]:


x


# In[44]:


y


# In[45]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x


# In[46]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)


# In[47]:


xtrain


# In[48]:


#machine learning algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain) #input and output training data
predict=knn.predict(xtest)
predict


# In[49]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict)


# In[50]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)#input and output training data
predict1=lr.predict(xtest)
predict1


# In[51]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict1)


# In[52]:


from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='adam',hidden_layer_sizes=(5,2),random_state=2,max_iter=2000)
clf.fit(xtrain,ytrain)#input and output training data
predict2=clf.predict(xtest)
predict2


# In[53]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict2)


# In[54]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,OneHotEncoder

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

models={
    "LogisticRegression":LogisticRegression(),
    "RandomForest":RandomForestClassifier(),
    "KNN":KNeighborsClassifier(),
    "SVM":SVC(),
    "GradientBoosting":GradientBoostingClassifier()
}

results={}

for name,model in models.items():
    pipe=Pipeline(
        [
            ('scaler',StandardScaler()),
            ('model',model)
        ]
    )

    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    results[name]=acc
    print(f"{name}Accuracy:{acc:4f}")
    print(classification_report(y_test,y_pred))


# In[55]:


import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='lightgreen')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks (rotation=45)
plt.grid(True)
plt.show()


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}
results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Saved best model as best_model.pkl")


# In[57]:


#get_ipython().run_line_magic('pip', 'install streamlit')


# In[58]:


#get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nimport joblib\n\n#Load the trained model\nmodel = joblib.load("best_model.pkl")\n\nst.set_page_config(page_title="Employee Salary Classification", page_icon="", layout="centered")\n\nst.title(" Employee Salary Classification App")\nst.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")\n\n#Sidebar inputs (these must match your training feature columns)\nst.sidebar.header("Input Employee Details")\n\n# Replace these fields with your dataset\'s actual input columns\nage = st.sidebar.slider("Age", 18, 65, 30)\neducation = st.sidebar.selectbox("Education Level", [\n    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"\n])\n\n\noccupation = st.sidebar.selectbox("Job Role", [\n    "Tech-support", "Craft-repair", "Other-service", "Sales",\n    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",\n    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",\n    "Protective-serv", "Armed-Forces"\n])\n\nhours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)\nexperience = st.sidebar.slider("Years of Experience", 0, 40, 5)\n\n# Build input DataFrame (must match preprocessing of your training data)\ninput_df = pd.DataFrame({\n    \'age\': [age], # \'age\' needs to be defined\n    \'education\': [education],\n    \'occupation\': [occupation],\n    \'hours-per-week\': [hours_per_week],\n    \'experience\': [experience]\n})\n\n\nst.write("### Input Data")\nst.write(input_df)\n\n#Predict button\nif st.button("Predict Salary Class"):\n    prediction = model.predict(input_df)\n    st.success(f" Prediction: {prediction[0]}")\n\n#Batch prediction\nst.markdown("---")\nst.markdown("#### Batch Prediction")\nuploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")\nif uploaded_file is not None:\n    batch_data = pd.read_csv(uploaded_file)\n    st.write("Uploaded data preview:", batch_data.head())\n    batch_preds = model.predict(batch_data)\n    batch_data[\'PredictedClass\'] = batch_preds\n    st.write("Predictions:")\n    st.write(batch_data.head())\n    csv = batch_data.to_csv(index=False).encode(\'utf-8\')\n    st.download_button("Download Predictions CSV", csv, file_name=\'predicted_classes.csv\', mime=\'text/csv\')\n')


# In[59]:


#get_ipython().run_line_magic('pip', 'install streamlit pyngrok')


# In[60]:

import os
os.system('ngrok authtoken 30EkOmbcInWye6cJ6p28SQuweKX_3Mz4edYiKnuHv2SuaD4ZH')


# In[61]:


import os
import threading

def run_streamlit():
            os.system('streamlit run app.py--server.port 8501')

thread = threading. Thread (target=run_streamlit)
thread.start()


# In[62]:


from pyngrok import ngrok
import time

time.sleep(5)
public_url = ngrok.connect(8501)
print("Your Streamlit app is live here:", public_url)


# In[63]:
os.system('!streamlit run Employee_Salary_Prediction.py')
