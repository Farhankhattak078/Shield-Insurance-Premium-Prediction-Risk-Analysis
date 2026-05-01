import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_excel('dataset/premiums_with_life_style.xlsx')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

df1 = df[df.age<=100]
df1 = df1[df1['number_of_dependants'] <= 10]

quantile_threeshold = df1.income_lakhs.quantile(0.99)
df1 = df1[df1['income_lakhs']<=quantile_threeshold]

df1['income_level'] = df1['income_level'].map({
    '25L - 40L':1,
    '10L - 25L':2,
    '<10L':3,
    '> 40L':4
})

df1['insurance_plan'] = df1['insurance_plan'].map({
    'Bronze' : 1,
    'Silver' : 2,
    'Gold' : 3
})

risk_score = {
    'high blood pressure': 6,
    'no disease': 0,
    'heart disease': 8,
    'thyroid': 5,
    'diabetes': 6,
    'none': 0
}
df1['medical_history'] = df1['medical_history'].str.lower().str.strip()
df1[['disease1', 'disease2']] = df1['medical_history'].str.split('&', expand=True)
df1['disease1'] = df1['disease1'].str.strip().fillna('none')
df1['disease2'] = df1['disease2'].str.strip().fillna('none')
df1['total_risk_score'] = (
    df1['disease1'].map(risk_score).fillna(0) +
    df1['disease2'].map(risk_score).fillna(0)
)
max_score = df1['total_risk_score'].max()
min_score = df1['total_risk_score'].min()

print("Max Score:", max_score, "Min Score:", min_score)

cols_to_scale = ['age','number_of_dependants','income_lakhs','income_level','insurance_plan']
scaler = MinMaxScaler()
scaler.fit(df1[cols_to_scale])

joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to model/scaler.pkl")
