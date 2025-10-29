import pandas as  pd

df = pd.read_csv('SMSSpamCollection.txt', sep ='\t' , names=['lable' , 'text'])

df.to_csv('data/emails.csv',index=False)