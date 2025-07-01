# preprocessing.py

import pandas as pd
import numpy as np
import ast
# Note: LabelEncoder is NOT imported here; it will be loaded from .pkl and passed
# into preprocess_for_prediction from your app.py file.

# --- Helper Functions (Copied from your notebook) ---

def extract_education_fields(edu_str):
    try:
        # Handle scalar NaN or None first
        if pd.isna(edu_str): return []
        # Handle string '[]'
        if isinstance(edu_str, str) and edu_str == '[]': return []
        
        if isinstance(edu_str, str):
            try:
                edu_list = ast.literal_eval(edu_str)
            except (ValueError, SyntaxError):
                return [] 
        elif isinstance(edu_str, list):
             edu_list = edu_str
        else:
             return [] 

        return [edu.get('field', '') for edu in edu_list if isinstance(edu, dict) and edu.get('field')]
    except Exception as e:
        return []

def extract_education_degrees(edu_str):
    try:
        # Handle scalar NaN or None first
        if pd.isna(edu_str): return []
        # Handle string '[]'
        if isinstance(edu_str, str) and edu_str == '[]': return []

        if isinstance(edu_str, str):
           try:
               edu_list = ast.literal_eval(edu_str)
           except (ValueError, SyntaxError):
               return [] 
        elif isinstance(edu_str, list):
            edu_list = edu_str
        else:
            return [] 

        return [edu.get('degree', '') for edu in edu_list if isinstance(edu, dict) and edu.get('degree')]
    except Exception as e:
        return []

edu_level_map = {
    'bachelor': 1,'master': 2,'phd': 3,'doctor': 3,'mba': 2,
    'postgraduate': 2,'certificate': 0.5,'diploma': 0.5
}
def get_highest_education(degrees):
    highest = 0
    if not isinstance(degrees, list): return 0
    for degree in degrees:
        degree_lower = str(degree).lower()
        for key, value in edu_level_map.items():
            if key in degree_lower and value > highest: highest = value
    return highest

def has_education_in_domain(fields, domain_keywords):
    if not fields: return 0 
    if not isinstance(fields, list): return 0
    for field in fields:
        field_lower = str(field).lower()
        for keyword in domain_keywords:
            if keyword in field_lower: return 1
    return 0

tech_keywords = ['computer', 'software', 'information tech', 'data', 'programming', 'web']
business_keywords = ['business', 'management', 'marketing', 'finance', 'economics', 'accounting']
engineering_keywords = ['engineering', 'mechanical', 'electrical', 'civil', 'chemical']
science_keywords = ['science', 'biology', 'chemistry', 'physics', 'mathematics']
arts_keywords = ['art', 'design', 'music', 'literature', 'philosophy', 'language']

def has_skill_mentions(text, skill_keywords):
    # This function expects 'text' to be a scalar string or NaN
    if pd.isna(text) or text == '[]' or text == '': 
        return 0
    text_lower = str(text).lower()
    for keyword in skill_keywords:
        if keyword in text_lower: return 1
    return 0

tech_skills = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'data analysis', 'machine learning', 'ai', 'cloud', 'aws', 'azure', 'devops']
business_skills = ['leadership', 'management', 'strategy', 'marketing', 'sales', 'project management', 'communication', 'negotiation']
creative_skills = ['design', 'photoshop', 'illustrator', 'adobe', 'creative', 'writing']

# THIS IS THE ONLY count_languages FUNCTION YOU SHOULD HAVE:
def count_languages(langs_input):
    # This function receives the value from the 'languages' cell,
    # which could be a list (from JSON), a string '[]' (from defaults), or pd.NA/None.

    # print(f"Debug count_languages received: {langs_input} (type: {type(langs_input)})") # Optional debug

    # Case 1: Input is pd.NA, NaN, or None (scalar check)
    # Check if it's a scalar missing value. hasattr check avoids error if langs_input is list/array.
    if not hasattr(langs_input, '__iter__') and pd.isna(langs_input):
        # print("Debug count_languages: scalar NaN/None, returning 0")
        return 0

    # Case 2: Input is the literal string '[]'
    if isinstance(langs_input, str) and langs_input == '[]':
        # print("Debug count_languages: input is string '[]', returning 0")
        return 0

    # Case 3: Input is an actual list (e.g., from JSON parsing like your Postman data)
    if isinstance(langs_input, list):
        # print(f"Debug count_languages: input is list, returning length: {len(langs_input)}")
        return len(langs_input)

    # Case 4: Input is a string that might represent a list and needs to be evaluated
    if isinstance(langs_input, str):
        try:
            langs_list_eval = ast.literal_eval(langs_input)
            if isinstance(langs_list_eval, list):
                # print(f"Debug count_languages: evaluated string to list, returning length: {len(langs_list_eval)}")
                return len(langs_list_eval)
            else:
                # print("Debug count_languages: evaluated string was not a list, returning 0")
                return 0 
        except (ValueError, SyntaxError):
            # print("Debug count_languages: string could not be evaluated as list, returning 0")
            return 0 
        except Exception: 
            # print(f"Debug count_languages: Unexpected error during ast.literal_eval on string input")
            return 0
            
    # Fallback for other unexpected types (e.g. if it's a numpy array or pandas Series that slipped through)
    # This path should ideally not be hit if inputs are handled correctly before .apply
    # print(f"Debug count_languages: input type '{type(langs_input)}' not directly handled, returning 0")
    return 0

# --- NEW Function for API Prediction ---

def preprocess_for_prediction(input_data, region_encoder):
    """
    Preprocesses raw input data (dict) for prediction using loaded encoders.
    ... (docstring) ...
    """
    try:
        # 1. Create a single-row DataFrame
        expected_cols = [
            'education', 'about', 'region', 'following', 'recommendations_count',
            'languages', 'position', 'current_company:name', 'posts', 'groups',
            'experience', 'people_also_viewed', 'educations_details',
            'certifications', 'recommendations', 'volunteer_experience', 'сourses'
        ]
        defaults = {
            'current_company:name': 'Unknown', 'posts': '[]', 'groups': '[]',
            'experience': '[]', 'people_also_viewed': '[]', 'educations_details': 'Unknown',
            'education': '[]', 'languages': '[]', 'certifications': '[]',
            'recommendations': '[]', 'volunteer_experience': '[]', 'сourses': '[]',
            'recommendations_count': 0, 'following': 0, 'region': 'Unknown',
            'about': '', 'position': 'Unknown'
        }

        df_row_data = {}
        for col in expected_cols:
            df_row_data[col] = input_data.get(col, defaults.get(col))

        processed_df = pd.DataFrame([df_row_data])

        # Ensure correct types after fillna/creation
        processed_df['following'] = processed_df['following'].astype(float)
        processed_df['recommendations_count'] = processed_df['recommendations_count'].astype(float)
        processed_df['region'] = processed_df['region'].astype(str)
        # 'languages' and 'education' values are passed as-is to .apply,
        # their respective helper functions will handle their types (list or string '[]')

        # 2. Apply Feature Engineering (using helper functions)
        processed_df['has_education'] = processed_df['education'].apply(lambda x: 0 if (isinstance(x, str) and x == '[]') or (isinstance(x, list) and not x) else 1)
        processed_df['education_fields'] = processed_df['education'].apply(extract_education_fields)
        processed_df['education_degrees'] = processed_df['education'].apply(extract_education_degrees)
        processed_df['highest_education'] = processed_df['education_degrees'].apply(get_highest_education)
        processed_df['has_tech_education'] = processed_df['education_fields'].apply(lambda x: has_education_in_domain(x, tech_keywords))
        processed_df['has_business_education'] = processed_df['education_fields'].apply(lambda x: has_education_in_domain(x, business_keywords))
        processed_df['has_engineering_education'] = processed_df['education_fields'].apply(lambda x: has_education_in_domain(x, engineering_keywords))
        processed_df['has_science_education'] = processed_df['education_fields'].apply(lambda x: has_education_in_domain(x, science_keywords))
        processed_df['has_arts_education'] = processed_df['education_fields'].apply(lambda x: has_education_in_domain(x, arts_keywords))
        processed_df['has_tech_skills'] = processed_df['about'].apply(lambda x: has_skill_mentions(x, tech_skills))
        processed_df['has_business_skills'] = processed_df['about'].apply(lambda x: has_skill_mentions(x, business_skills))
        processed_df['has_creative_skills'] = processed_df['about'].apply(lambda x: has_skill_mentions(x, creative_skills))
        
        processed_df['language_count'] = processed_df['languages'].apply(count_languages)

        # 3. Encode Region using the LOADED encoder
        input_region = processed_df['region'].iloc[0] 
        try:
            encoded_region = region_encoder.transform([input_region]) 
        except ValueError:
            print(f"Warning: Region '{input_region}' not seen during training. Encoding as 'Unknown'.")
            try:
                 encoded_region = region_encoder.transform(['Unknown'])
            except ValueError:
                 print("Error: Default 'Unknown' region not found in encoder. Using 0 as fallback.")
                 encoded_region = np.array([0]) 

        processed_df['region_encoded'] = encoded_region[0]

        # 4. Select Final Features for the Model
        features_for_model = [
            'has_education', 'highest_education',
            'has_tech_education', 'has_business_education', 'has_engineering_education',
            'has_science_education', 'has_arts_education',
            'has_tech_skills', 'has_business_skills', 'has_creative_skills',
            'recommendations_count', 'following', 'language_count',
            'region_encoded'
        ]

        missing_model_features = [col for col in features_for_model if col not in processed_df.columns]
        if missing_model_features:
             print(f"Error: Model feature columns missing after processing: {missing_model_features}")
             return None

        X_final = processed_df[features_for_model]

        return X_final

    except Exception as e:
        print(f"Error during preprocessing for prediction: {e}")
        import traceback
        traceback.print_exc()
        print(f"Input data that may have caused error: {input_data}")
        return None
