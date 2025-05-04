from pandas import DataFrame

def add_week_day(fraude: DataFrame) -> DataFrame:
    fraude = fraude.copy()
    fraude['week_day'] =  fraude['date'].apply(lambda date: date.strftime("%A"))
    return fraude

def split_city(fraude: DataFrame) -> DataFrame:
    fraude = fraude.copy()
    country_city = fraude['city'].str.split(',')
    fraude['country'] = country_city.apply(lambda x: x[1] if len(x) > 1 else None)
    fraude['city'] = country_city.apply(lambda x: x[0] if len(x) > 0 else None)
    return fraude

def get_features():
    return ['amount','type']

def get_target():
    return 'isfraud'