import streamlit as st 
import pandas as pd 
import numpy as np 


import pickle
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
import time
from sklearn import model_selection

n=193
Pkl_Filename="C:/Users/hp/Desktop/MyModel.pkl"
with open(Pkl_Filename, 'rb') as file: model = pickle.load(file)
X1 = pd.DataFrame()
y1 = pd.DataFrame()
def main():
	
	#if st.Button('Open a Dataset")
	#activities = ["Open Data Set","Data Visualizations","Prediction","About"]	
	#choice = st.sidebar.selectbox("Select Activities",activities)
		"""Predictive Maintenance System """
		#if st.checkbox("Click here to Upload Data Set"):
		st.title('MHS')
		st.subheader("Click browse file option in the center to upload a data set")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			X1=df
			y1=df
				#df=df.drop(["remaining_cycle"],axis=1)
				#my_bar = st.progress(0)
				#for p in range(50):
				#time.sleep(0.1)
				#my_bar.progress(p + 5)
				
			if st.checkbox("Upload Data Set"):
					st.dataframe(X1.head(193))
					all_columns = df.columns.to_list()
					# Radio Buttons
					status = st.radio("Select an option",("Rows/Col","Columns List","Statistics"))
					if status == 'Rows/Col':
						st.write('Numbers of Rows and Col found',df.shape)
					elif status=="Columns List":
						st.write(all_columns)
					else:
						st.write(df.describe().transpose())
	
	
			if st.checkbox("Select features"):
				st.subheader("Click [Choose an option] below to select Features ")
				selected_columns = st.multiselect("Select features",all_columns)
				X1 = df[selected_columns]
		
			
			if st.checkbox("Select Target"):
						st.subheader("Click [Choose an option] below to select  Target varible ")
						selected_columns2 = st.multiselect("Select Target",all_columns)
						y1=df[selected_columns2]
			

			if st.checkbox("Prediction from Pretrained Model"):
				st.subheader("Please scroll down to see the results")
				y=y1
				X=X1
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
				#st.write(X_train)
				#X_train=X_train.reshape(-1, 1)
				model.fit(X_train, y_train.values.ravel());
				predictedLife = model.predict(X_test)
				score = accuracy_score(predictedLife, y_test)
				
				# Radio Buttons
				status2 = st.radio("Select an option",("Accuracy","Confusion Matrix","Summary"))
				if status2 == 'Accuracy':
					st.write('Accuracy of Prediction : ', int(score*100) , '%')
				elif status2=="Summary":
					st.write('Confusion Matrix : \n', confusion_matrix(predictedLife, y_test))
				else:
					st.write('Summary \n\n', classification_report(predictedLife, y_test))


				
				
		
			if st.checkbox("Start Simulation for validation"):	
				#X_test.reset_index(inplace=True)
				entry = X_test.head(n).values
				#entry.reset_index(inplace=True)
				index = list()
				#st.write(n)
				for i in range(0,n):
					index.append(X_test.head(n).index[i])
				for i in range(0,n):
					if i == n-1:
							break
					st.dataframe(X_test[i:i+1])
					#time.sleep(0.15)
					val1 = index[i+1]
					predictedValue = model.predict(entry)
					RealValue = y_test.loc[val1]
					#st.write('-------------')
					#st.write('Real Value : ', RealValue)
					#st.write('Predicted Value : ', predictedValue[i])
					if predictedValue[i]:
						st.warning("20 or less second remaining in failure")
						#time.sleep(0.50)
					else:
						st.success("No chance of failure")	

							#X = X1
							#y=y1
							#seed = 7
							#models = []
							#models.append(('LR', LogisticRegression()))
							#models.append(('LDA', LinearDiscriminantAnalysis()))
							#models.append(('KNN', KNeighborsClassifier()))
							#models.append(('CART', DecisionTreeClassifier()))
							#models.append(('NB', GaussianNB()))
							#models.append(('SVM', SVC()))
							#model_names = []
							#model_mean = []
							#model_std = []
							#all_models = []
							#scoring = 'accuracy'
							#for name, model in models:
								#kfold = model_selection.KFold(n_splits=10, random_state=seed)
								#cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
								#model_names.append(name)
								#model_mean.append(cv_results.mean())
								#model_std.append(cv_results.std())
								#accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
								#all_models.append(accuracy_results)
		
							#st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))	


		
if __name__ == '__main__':
	main()