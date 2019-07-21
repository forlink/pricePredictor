
from pricePredictor import classifier
import pandas as pd

if __name__=='__main__':
    df=pd.read_csv('EUR_USD.csv',index_col=0)
    clf=classifier()
    clf.train(df,verbose=True)    
    result=clf.predict(df)
    if result==-1:
        print('Sell!')
    if result==1:
        print('Buy!')