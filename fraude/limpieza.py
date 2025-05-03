from pandas import DataFrame
import pandas as pd

def clean_column_name(column_name: str) -> str:
    column_name = column_name.lower()
    column_name = column_name.replace(" ", "_")
    column_name = column_name.replace("(", "")
    column_name = column_name.replace(")", "")
    return column_name
    

def clean_column_names(fraude: DataFrame) -> DataFrame:
    fraude = fraude.copy()
    fraude = fraude.rename(columns={name: clean_column_name(name) for name in fraude.columns})
    return fraude

def fix_datetime_columns(fraude: DataFrame) -> DataFrame:
    fraude = fraude.copy()
    months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
    months = {month: number for month, number in zip(months, range(1,13))}
    fraude['date'] = fraude['date'].str.lower().str.split('-').apply(lambda x: f"{2000+int(x[2])}/{months[x[1]]}/{x[0]}" if len(x) > 2 and x[1] in months else None)
    fraude['date']= pd.to_datetime(fraude['date'], yearfirst=True)
    return fraude

def remove_outliers(fraude: DataFrame) -> DataFrame:
    Q1 = fraude['amount'].quantile(0.25)
    Q3 = fraude['amount'].quantile(0.75)
    IQR = Q3 - Q1
    fraude = fraude[~((fraude['amount'] < (Q1 - 1.5 * IQR)) | (fraude['amount'] > (Q3 + 1.5 * IQR)))]
    return fraude

def remove_zeros(fraude: DataFrame) -> DataFrame:
    fraude = fraude.query('amount > 0')
    return fraude