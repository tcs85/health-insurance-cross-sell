import pickle 
import pandas as pd  
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

class HealthInsurance:
    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaler = pickle.load(open('/features/annual_premium_scaler.pkl', 'rb'))
        self.age_scalar = pickle.load(open('/features/age_scaler.pkl', 'rb'))
        self.vintage_scalar = pickle.load(open('/features/vintage_scaler.pkl', 'rb'))
        self.target_encode_gender = pickle.load(open('/features/target_encode_gender.pkl', 'rb'))
        self.target_encode_region_code = pickle.load(open('/features/target_encode_region_code.pkl', 'rb'))
        self.fe_policy_sales_channel = pickle.load(open('/features/fe_policy_sales_channel.pkl', 'rb'))
        
    def data_cleaning(self, df1):
        ## 1.0 Rename Columns
        cols_new = ['id', 'gender', 'age', 'region_code', 'policy_sales_channel', 'driving_license', 'vehicle_age', 'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage', 'response']
        df1.columns = cols_new
        return df1
        
    def feature_engineering(self, df2):
        ## 2.0 Feature Engineering
        # vehicle age
        df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over 2 years' if x == '> 2 Years' 
                                                          else 'between 1-2 years' if x == '1-2 Year' 
                                                          else 'under 1 year')
        # vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        
        return df2
    
    def data_preparation(df5):
        # 3.0 Preprocessing
        # Annual Premium - Standard Scaler
        df5['annual_premium'] = self.annual_premium_scaler.transform(df5[['annual_premium']].values)
        
        # Age - Min Max Scaler
        df5['age'] = self.age_scalar.transform(df5[['age']].values)
        
        # Vintage - Min Max Scaler
        df5['vintage'] = self.vintage_scalar.transform(df5[['vintage']].values)
        
        # gender - Target Encoding
        df5.loc[:, 'gender'] = df5.loc[:, 'gender'].map(self.target_encode_gender)
        
        # region_code - Target Encoding
        df5.loc[:, 'region_code'] = df5.loc[:, 'region_code'].map(self.target_encode_region_code)
        
        # policy_sales_channel - Frequency Encoding
        df5.loc[:, 'policy_sales_channel'] = df5.loc[:, 'policy_sales_channel'].map(self.fe_policy_sales_channel)
        
        # Vehicle Age
        df5 = pd.get_dummies(df5, prefix=['vehicle_age'], columns=['vehicle_age'])
        
        # Reset Index
        df5 = df5.reset_index(drop=True)
                
        # Featuring Selection
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'policy_sales_channel', 'previously_insured', 'vehicle_damage', 'vehicle_age_over 2 years', 'vehicle_age_under 1 year']
    
        return df5 [cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict_proba(test_data)
        
        # join pred into the original data
        original_data['score'] = pred
        
        return original_data.to_json(orient='records', date_format='iso')
    