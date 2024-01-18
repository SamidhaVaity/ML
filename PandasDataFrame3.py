import pandas as pd
import numpy as np

df1 = {'One':pd.Series([1,2,3],index = ['a','b','c']),
    'two':pd.Series([1,2,3],index = ['a','b','c'])}

df2 = {'One':pd.Series([1,2,3],index = ['a','b','c']),
    'two':pd.Series([1,2,3],index = ['a','b','c'])}

data = {'Item1':df1,'Item2':df2}

p = pd.Panel(data)
print(p)