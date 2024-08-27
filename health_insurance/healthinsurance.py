import pickle 
import pandas as pd  
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

class HealthInsurance:
    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaler     = pickle.load(open('features/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler                = pickle.load(open('features/age_scaler.pkl', 'rb'))
        self.vintage_scaler            = pickle.load(open('features/vintage_scaler.pkl', 'rb'))
        self.target_encode_gender      = pickle.load(open('features/target_encode_gender.pkl', 'rb'))
        self.target_encode_region_code = pickle.load(open('features/target_encode_region_code.pkl', 'rb'))
        self.fe_policy_sales_channel   = pickle.load(open('features/fe_policy_sales_channel.pkl', 'rb'))
        # self.target_vehicle_age        = pickle.load(open('features/target_vehicle_age.pkl', 'rb'))
        
    def data_cleaning(self, df1):
        ## 1.0 Rename Columns
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 
                    'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage', 'response']
        df1.columns = cols_new
        return df1
        
    def feature_engineering(self, df2):
        ## 2.0 Feature Engineering
        # vehicle age
        # df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year' )
        # vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        
        return df2
    
    def data_preparation(self, df5):
        # 3.0 Preprocessing
        # anual premium - StandarScaler
        df5['annual_premium'] = self.annual_premium_scaler.transform( df5[['annual_premium']].values )

        # Age - MinMaxScaler
        df5['age'] = self.age_scaler.transform( df5[['age']].values )

        # Vintage - MinMaxScaler
        df5['vintage'] = self.vintage_scaler.transform( df5[['vintage']].values )

        # gender - One Hot Encoding / Target Encoding
        df5.loc[:, 'gender'] = df5['gender'].map( self.target_encode_gender )

        # region_code - Target Encoding / Frequency Encoding
        df5.loc[:, 'region_code'] = df5['region_code'].map( self.target_encode_region_code )

        # vehicle_age - One Hot Encoding / Frequency Encoding
        # df5 = pd.get_dummies( df5, prefix='vehicle_age', columns=['vehicle_age'] )
        # df5.loc[:, 'vehicle_age'] = df5['vehicle_age'].map(target_encode_vehicle_age)
        
        # policy_sales_channel - Target Encoding / Frequency Encoding
        df5.loc[:, 'policy_sales_channel'] = df5['policy_sales_channel'].map( self.fe_policy_sales_channel )
        
        # Feature Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured',
                         'policy_sales_channel']
        
        
        return df5 [cols_selected].dropna()
    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
    