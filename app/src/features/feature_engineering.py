
from sklearn import preprocessing

def feature_engineering(train_df, test_df):
    """
        Función para encapsular la tarea de ingeniería de variables

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """
    train_df = create_domain_knowledge_features(train_df)
    test_df = create_domain_knowledge_features(test_df)

    return train_df.copy(), test_df.copy()


def create_domain_knowledge_features(df):
    """
        Función la creación de variables de contexto

        Args:
           df (DataFrame):  Dataset.
        Returns:
           DataFrame. Dataset.
    """
    # creación de variable Child de tipo booleana
    #  df['Child'] = 0
    #  df.loc[df.Age < 16, 'Child'] = 1
   
    df['gender'] = df['gender'].map({'m': 0, 'f': 1})
    df['jaundice'] = df['jaundice'].map({'no': 0, 'yes': 1})
    df['austim'] = df['austim'].map({'no': 0, 'yes': 1})
    df['used_app_before'] = df['used_app_before'].map({'no': 0, 'yes': 1})
    df['ethnicity'] = df['ethnicity'].replace({'others': 'Others', '?': 'Others'})
    df['PuntuacionTest'] = df['A1_Score'] + df['A2_Score'] + df['A3_Score'] + df['A4_Score'] + df['A5_Score'] + df['A6_Score'] + df['A7_Score'] + df['A8_Score'] + df['A9_Score'] + df['A10_Score'] 
    label_encoder = preprocessing.LabelEncoder()
    df['ethnicity']= label_encoder.fit_transform(df['ethnicity']) 
    df['contry_of_res']= label_encoder.fit_transform(df['contry_of_res']) 


    return df.copy()
