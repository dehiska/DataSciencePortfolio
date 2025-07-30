import pandas as pd
import numpy as np
import seaborn as sns
import ast
import holidays
from uszipcode import SearchEngine
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def impute_missing_values(df, ignore_columns=None):
    """
    Imputes missing values:
    - For float or numeric columns, fills with mean.
    - For categorical or other columns, fills with mode.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        ignore_columns (list or None): List of columns to skip. Default is None.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df = df.copy()
    if ignore_columns is None:
        ignore_columns = []

    for col in df.columns:
        if col in ignore_columns:
            continue
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
            else:
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)
    return df


def cleaning(df):
    colnames_to_int = ['marital_status', 'high_education_ind', 'address_change_ind', 'policy_report_filed_ind']
    df[colnames_to_int] = df[colnames_to_int].astype(int)

    df['witness_present_ind'] = ["NP" if x == 0 else
                                 "P" if x == 1 else
                                 "DK" for x in df['witness_present_ind']]

    colnames_to_str = ['witness_present_ind','zip_code']
    df[colnames_to_str] = df[colnames_to_str].astype(str)

    df['claim_date']=pd.to_datetime(df['claim_date'])
    df['zip_code'] = df['zip_code'].str.zfill(5)

    return(df)

def age_cap(df, age_cap, age_col='age_of_driver'):
    df.loc[df[age_col] > age_cap, age_col] = age_cap
    return df

def assign_age_group(df, age_col='age_of_driver', new_col='age_group'):
    """
    Adds a categorical age group column to the dataframe based on age_of_driver.
    Groups:
        - '18-19'
        - '20-38'
        - '39-49'
        - '50-81'
        - '82+'
    """
    bins = [17, 19, 38, 49, 81, float('inf')]
    labels = ['18-19', '20-38', '39-49', '50-81', '82+']
    df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=True).astype('str')
    return df


def extract_datetime_features(df, date_col='claim_date', include_holidays=True):
    """
    Extracts basic datetime features from a given datetime column.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        date_col (str): The name of the datetime column.

    Returns:
        pd.DataFrame: DataFrame with new datetime-derived columns.
    """
    df = df.copy()

    # Ensure column is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Extract common date features
    df[f'{date_col}.year'] = df[date_col].dt.year.astype('str')
    df[f'{date_col}.month'] = df[date_col].dt.month.astype('str')
    df[f'{date_col}.day'] = df[date_col].dt.day.astype('str')
    df[f'{date_col}.dayofweek'] = df[date_col].dt.dayofweek.astype('str')
    df[f'{date_col}.weekofyear'] = df[date_col].dt.isocalendar().week.astype('str')

    # Additional datetime features
    df[f'{date_col}.quarter'] = df[date_col].dt.quarter.astype('str')
    df[f'{date_col}.is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)

    if include_holidays:
        us_holidays = holidays.US(years=[2015, 2016])
        holiday_dates = pd.to_datetime(list(us_holidays.keys()))

        # Expand to ±2 days around each holiday
        expanded_dates = set()
        for date in holiday_dates:
            for offset in range(-2, 3):  # -2, -1, 0, +1, +2
                expanded_dates.add(date + pd.Timedelta(days=offset))

        df[f'{date_col}.near_holiday'] = df[date_col].isin(expanded_dates).astype(int)

    # The following line was removed to keep 'claim_date' in the DataFrame:
    # df = df.drop(columns = ['claim_date', 'claim_day_of_week'])

    return df


def process_zipcode_features(df, zip_col='zip_code', plot=False):
    """
    Looks up and processes ZIP code-level features for a given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a ZIP code column.
        zip_col (str): Name of the ZIP code column in the DataFrame.
        plot (bool): If True, plots histogram of log population.

    Returns:
        pd.DataFrame: Processed ZIP code DataFrame with selected features.
    """
    search_zip = SearchEngine()
    unique_zip = df[zip_col].unique()

    zip_code_basic_features = ['zipcode', 'zipcode_type', 'state', 'population']
    zip_code_lite = []

    for zip_code in unique_zip:
        zip_info = search_zip.by_zipcode(zip_code)
        if zip_info is not None:
            zip_dict = zip_info.to_dict()
            zip_dict_lite = {key: zip_dict.get(key, np.nan) for key in zip_code_basic_features}
            zip_code_lite.append(zip_dict_lite)

    # Handle ZIP code 0 case explicitly
    zero_dict_lite = {key: np.nan for key in zip_code_basic_features}
    zero_dict_lite['zipcode'] = '00000'
    zero_dict_lite['zipcode_type'] = 'UNIQUE'
    zip_code_lite.append(zero_dict_lite)

    zip_code_df = pd.DataFrame(zip_code_lite)
    zip_code_df['zipcode'] = zip_code_df['zipcode'].astype(str).str.zfill(5)

    # Compute log population
    zip_code_df['log_population'] = np.log1p(zip_code_df['population'])

    # Bin log population
    zip_code_df['log_pop_bin'] = 0
    bins = [0, 5, 10, np.inf]
    labels = [1, 2, 3]
    non_null_mask = zip_code_df['log_population'].notnull()

    zip_code_df.loc[non_null_mask, 'log_pop_bin'] = pd.cut(
        zip_code_df.loc[non_null_mask, 'log_population'],
        bins=bins,
        labels=labels,
        right=False
    ).astype(int)

    # Optional histogram
    if plot:
        sns.histplot(zip_code_df['log_population'], bins=15)

    zip_features = zip_code_df.drop(columns=['population', 'log_population'])

    df = df.merge(zip_features, left_on='zip_code', right_on='zipcode', how='left')
    df = df.drop(columns= ['zip_code', 'zipcode'])
    # Clean up
    return df

def price_categories(df, col = 'vehicle_price', new_col_name = 'vehicle_price_categories'):
    df = df.copy()
    df[new_col_name] = ['under_15k' if x<=15000 else
                        'btw_20_30k' if 20000<=x<30000 else
                        'btw_30_40k' if 30000<=x<40000 else
                        'btw_40_50k' if 40000<=x<50000 else
                        'above_50k' for x in df[col]]
    return(df)

def liab_prct_group(df, col = 'liab_prct', new_col_name = 'liab_prct_group'):
    bins = [0, 5, 47.5, 52.5, 95, np.inf]
    labels = [0, 1, 2, 3, 4]

    df[new_col_name] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return(df)


def add_features(df, age_cap_value=82, exclude_vars='witness_present_ind', include_holidays=True):
    """
    Runs the full preprocessing pipeline on the input dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        age_cap_value (int): Cap value for age_of_driver.
        exclude_vars (list): List of columns to exclude from imputation (default: None).
        include_holidays (bool): Whether to include holiday-based datetime features.

    Returns:
        pd.DataFrame: Fully processed dataframe.
    """
    df_processed = df.copy()

    df_processed = impute_missing_values(df_processed, exclude_vars)
    df_processed = cleaning(df_processed)
    df_processed = age_cap(df_processed, age_cap_value)
    df_processed = assign_age_group(df_processed)
    df_processed = extract_datetime_features(df_processed, include_holidays=include_holidays)
    df_processed = process_zipcode_features(df_processed)
    df_processed = price_categories(df_processed)
    df_processed = liab_prct_group(df_processed) # This function was missing from the add_features call previously

    return df_processed

def drop_ignored_columns(df, ignore_var):
    """
    Returns a DataFrame with columns from ignore_var removed (if they exist).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        ignore_var (list): List of column names to ignore/remove.

    Returns:
        pd.DataFrame: DataFrame with ignored columns dropped.
    """
    # Keep only columns NOT in ignore_var
    filtered_cols = [col for col in df.columns if col not in ignore_var]
    return df[filtered_cols]


def fit_regular_transformer(train_df, ignore_suffix='_count'):
    # Identify regular columns
    regular_cols = [col for col in train_df.columns if not col.endswith(ignore_suffix)]

    # Split regular into categorical and numerical
    categorical_cols = train_df[regular_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = train_df[regular_cols].select_dtypes(include=['number']).columns.tolist()
    if 'claim_number' in numerical_cols:
        numerical_cols.remove('claim_number')

    # Initialize transformers
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()

    # Fit transformers
    onehot.fit(train_df[categorical_cols])
    scaler.fit(train_df[numerical_cols])

    # print(f"Fitted on {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns.")

    return onehot, scaler, categorical_cols, numerical_cols

def transform_regular_set(df, onehot, scaler, categorical_cols, numerical_cols):
    # Transform categorical
    cat_transformed = onehot.transform(df[categorical_cols])
    cat_df = pd.DataFrame(cat_transformed, columns=onehot.get_feature_names_out(categorical_cols), index=df.index)

    # Transform numerical
    num_transformed = scaler.transform(df[numerical_cols])
    num_df = pd.DataFrame(num_transformed, columns=numerical_cols, index=df.index)

    # Combine transformed parts
    transformed_df = pd.concat([num_df, cat_df], axis=1)

    # print(f"Transformed set shape: {transformed_df.shape}")
    return transformed_df