# pricePredictor

The pricePredictor class uses machine learning to predict forex or stock price changes.
In order to use the pricePredictor you have to provide a Pandas DataFrame containing columns "low","high","close","open" and a datetime index.

The train() method accepts a DataFrame in input and trains a classifier based on support vector machine (sklearn.LinearSVC).
The main steps of this method are:
1. Creation of features
The features are created in this way:
- compute a column of average prices prices_m that are equal to (high+low+close)/3
- compute for each row of the DataFrame the price variation with respect to the previous row as prices_m(i)/prices_m(i-1)-1
- compute for each row of the DataFrame a vector of the last 5 price variations and set them as the features
2. Creation of labels
- compute a column of average prices prices_m that are equal to (high+low+close)/3
- compute for each row of the DataFrame the variation of the price_m with respect to the next row prices_m(i+1)/prices_m(i)-1, and create a column with binary values: -1 if the price change is negative and 1 if the price change is positive
- set the last vector as the labels vector
3. Training
- Provide the features and labels just created to the LinearSVC.fit() method

The predict() method accepts a DataFrame in input with at least 5 rows and returns 1 if the price is supposed to increase in the next time unit or -1 if the price is supposed to decrease.
	
Launch demo.py for demonstration
