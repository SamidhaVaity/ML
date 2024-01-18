import pandas as pd

data = [{'Name':'PPA','Duration':'3','Fees':10500},{'Name':'Angular','Duration':'3','Fees':10500},
    {'Name':'Python','Fees':10500}]
df = pd.DataFrame(data)
print(df)

writer = pd.ExcelWriter('Marvellous.xlsx', engine = 'xlsxwriter')

df.to_excel(writer,sheet_name='sheet1')

writer.save()