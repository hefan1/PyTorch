import numpy
import pandas as pd

df=pd.read_csv('channelClusterResult.csv')

result=[[],[],[]]
for i in range(3):
    for j in range(512):
        result[i].append(1 if i==df.iloc[j][1] else 0)

print(result)
result=pd.DataFrame(data=result)
result.to_csv("GCLables.csv")