import pickle 
import pandas as pd  
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

class HealthInsurance:
    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaler = pickle.load(open('features/annual_premium_scaler.pkl', 'rb'))
        self.age_scalar = pickle.load(open('features/age_scaler.pkl', 'rb'))
        self.vintage_scalar = pickle.load(open('features/vintage_scaler.pkl', 'rb'))
        self.target_encode_gender = pickle.load(open('features/target_encode_gender.pkl', 'rb'))
        self.target_encode_region_code = pickle.load(open('features/target_encode_region_code.pkl', 'rb'))
        self.fe_policy_sales_channel = pickle.load(open('features/fe_policy_sales_channel.pkl', 'rb'))
        
    def data_cleaning(self, df1):
        ## 1.0 Rename Columns
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 
                    'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage', 'response']
        df1.columns = cols_new
        return df1
        
    def feature_engineering(self, df2):
        ## 2.0 Feature Engineering
        # vehicle age
        df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year' )
        # vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        
        return df2
    
    def data_preparation(self, df5):
        # 3.0 Preprocessing
        # Annual Premium - Standard Scaler
        df5['annual_premium'] = self.annual_premium_scaler.transform(df5[['annual_premium']].values)
        
        # Age - Min Max Scaler
        df5['age'] = self.age_scalar.transform(df5[['age']].values)
        
        # Vintage - Min Max Scaler
        df5['vintage'] = self.vintage_scalar.transform(df5[['vintage']].values)
        
        # gender - Target Encoding
        df5.loc[:, 'gender'] = df5.loc[:, 'gender'].map(self.target_encode_gender).astype('float64') 
        
        # region_code - Target Encoding
        df5.loc[:, 'region_code'] = df5.loc[:, 'region_code'].map(self.target_encode_region_code).astype('float64')
        
        # policy_sales_channel - Frequency Encoding
        df5.loc[:, 'policy_sales_channel'] = df5.loc[:, 'policy_sales_channel'].map(self.fe_policy_sales_channel).astype('float64')
        
        # Vehicle Age
        df5 = pd.get_dummies(df5, prefix=['vehicle_age'], columns=['vehicle_age'])
        
        # Featuring Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured',
                         'policy_sales_channel']
    
        return df5 [cols_selected]
    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )

        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
    