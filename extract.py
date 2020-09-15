import numpy as np
import pandas as pd
import preprocessors as pp

def run_extract():
    # read data files 
    data_lead = pd.read_csv("data/olist_marketing_qualified_leads_dataset.csv")
    data_leads_closed = pd.read_csv("data/olist_closed_deals_dataset.csv")

    # combine data and add leads converted label
    # Adding a column for leads closed or not 1==deal closed
    data = pd.merge(data_lead, data_leads_closed[['mql_id']], on='mql_id', how='left',indicator='closed')
    data['closed'] = np.where(data.closed == 'both', 1, 0)

    #split date into 3 separate features
    data.first_contact_date = pd.to_datetime(data.first_contact_date)
    data["contact_year"] = data.first_contact_date.dt.year
    data["contact_month"] = data.first_contact_date.dt.month
    data["contact_day"] = data.first_contact_date.dt.day


    # taking equal number of closed and not closed leads. closed(842) not closed(8000-842)
    data = data.sort_values('closed',ascending=False)
    data = data.iloc[:1684,:].copy()

    # read training data
    # data = pd.read_csv(config.TRAINING_DATA_FILE)
    data = pp.fill_missing_categorical(data,['origin'])
    data = pp.encode_categorical(data,['landing_page_id','origin'])
    data = pp.normalize(data,['landing_page_id', 'origin', 'contact_year', 'contact_month',
       'contact_day'])
    data = pp.drop_features(data,['mql_id', 'first_contact_date'])
    data.to_csv("data/dataset.csv")

    return data