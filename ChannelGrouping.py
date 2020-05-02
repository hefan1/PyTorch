from sklearn.cluster import MiniBatchKMeans

import numpy as np
import pandas as pd

df=pd.read_csv('channelPeakPos.csv')

X=np.array(df.iloc[0:,1:])

kmeans = MiniBatchKMeans(n_clusters=3,random_state=0,batch_size=2048,max_iter=200).fit_predict(X)#3

out=pd.DataFrame(data=kmeans)

out.to_csv('channelClusterResult.csv')