from  datetime import datetime, timedelta
import numpy as np, pandas as pd

CAL_DTYPES={"event_name_2": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

def prepare_tables(path):
    prices = pd.read_csv(path/"sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv(path/"calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
            
    return prices, cal

def create_event_map(cal, field):
    return {v: k for k, v in enumerate(cal[field].unique())}

def convert_to_type(df, cols, dt_type):
    for type_name in cols:
        df[type_name] = df[type_name].astype(dt_type)

def convert_uint8(df, cols):
    convert_to_type(df, cols, "uint8")
    
def convert_float16(df, cols):
    convert_to_type(df, cols, "float16")
    
def replace_cal_cols(cal):
    event_name_1_map = create_event_map(cal, 'event_name_1')
    cal.replace({'event_name_1': event_name_1_map}, inplace=True)
    event_type_1_map = create_event_map(cal, 'event_type_1')
    cal.replace({'event_type_1': event_type_1_map}, inplace=True)
    return event_name_1_map, event_type_1_map

day_of_year = 'Dayofyear'

def prepare_day_of_year(df):
    df[day_of_year] = getattr(df['date'].dt, day_of_year.lower()).astype('uint16')

def add_days_before(dt, day=25, month=12, col_name='before_christmas'):
    diff_list = []
    for d in dt['date']:
        target = datetime(d.year, month, day)
        diff = (target - d.to_pydatetime()).days
        if(diff < 0):
            christmas = datetime(d.year + 1, 12, 25)
            diff = (target - d.to_pydatetime()).days
        diff_list.append(diff)
    dt[col_name] = diff_list
    dt[col_name] = dt[col_name].astype('uint16')
    
state_map = {'CA': 0, 'TX': 1, 'WI': 2}
    
def create_dt(cal, prices, is_train = True, nrows = None, first_day = 1200, tr_last=1913, path=None):
    
    start_day = max(1, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(path/"sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols + ['state_id'], dtype = dtype)
    
    dt.replace({'state_id': state_map}, inplace=True)
    dt['state_id'] = dt['state_id'].astype("int16")
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols + ['state_id'],
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    dt.sort_values(['id', 'date'], inplace=True)
    prepare_day_of_year(dt)
    
    ## Dates
    
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "year": "year",
        "mday": "day",
    }
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
            
    uint8_types= ['month', 'wday', 'mday', 'week']
    convert_uint8(dt, uint8_types)
    
    ## Lag Price
    
    for lap_price in [1]:
        dt[f'lag_price_{lap_price}'] = dt[["id","sell_price"]].groupby("id")["sell_price"].shift(lap_price).astype('float16')

    convert_float16(dt, ['sales', "sell_price"])
    
    return dt