# data
ALL_LEADS_DATA = "data/olist_marketing_qualified_leads_dataset.csv"
LEADS_CLOSED_DATA = "data/olist_closed_deals_dataset.csv"
VALIDATION_DATA_FILE = "data/validation.csv"

TARGET = 'closed'

# input variables 
FEATURES = ['landing_page_id', 'origin', 'contact_year', 'contact_month',
       'contact_day']

# features to be droped
DROP_FEATURES = ['mql_id', 'first_contact_date']

# categorical variables to encode
CATEGORICAL_VARS = ['landing_page_id', 'origin']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['origin']