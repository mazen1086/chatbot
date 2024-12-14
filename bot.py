import os
import re
import json
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from kmodes.kmodes import KModes
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging
from datetime import datetime
import traceback
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load environment variables
load_dotenv()

def setup_logging():
    """Set up logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/churn_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Global logger
logger = setup_logging()

class CatBoostPipeline:
    def __init__(self, model_path= 'k_model.pkl', kmodes_path= 'cb_model.cbm'):
        self.catboost_model = None
        self.kmodes = None
        self.scalers = {}
        self.categorical_indices = None
        self.data_columns = [
            'gender', 'Senior_Citizen', 'Is_Married', 'Dependents', 'tenure', 'Dual',
            'Contract', 'Paperless_Billing', 'Payment_Method', 'Monthly_Charges',
             'Online_Security', 'Online_Backup', 'Device_Protection',
            'Streaming_TV', 'Streaming_Movies', 'Internet_Service', 'Tech_Support'
        ]
        self.numerical_features = ['tenure', 'Monthly_Charges']
        self.load_model(model_path, kmodes_path)

    def load_model(self, model_path = '/content/drive/MyDrive/eitesalat/catboost_model.cbm', kmodes_path ='/content/drive/MyDrive/eitesalat/models/kmods_model.pkl'):
        """Load the CatBoost and KModes models."""
        try:
            self.catboost_model = CatBoostClassifier()
            self.catboost_model.load_model(model_path)
            self.kmodes = joblib.load(kmodes_path)
        except FileNotFoundError:
            #st.error(f"Model files not found: {model_path} or {kmodes_path}")
            raise

    def preprocess_data(self, df):
        """Comprehensive data preprocessing pipeline."""
        data = self.validate_and_prepare_data(df)
        data = self.engineer_features(data)
        data_temp = self.encode_categorical(data)
        data['Cluster'] = self.apply_kmodes(data_temp)
        return self.finalize_features(data)

    def validate_and_prepare_data(self, df):
        """Validate and prepare input dataframe."""
        data = pd.DataFrame(columns=self.data_columns)
        for col in self.data_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            data[col] = df[col].copy()
        return data

    def engineer_features(self, data):
        """Advanced feature engineering."""
        #data['Senior_Citizen '] = data['Senior_Citizen '].map({0: 'No', 1: 'Yes'})
        
        # Combine related features
        data['O_B_S_Dp'] = data['Online_Security'] + "_" + data['Online_Backup'] + "_" + data['Device_Protection']
        data['S_TV_M'] = data['Streaming_TV'] + "_" + data['Streaming_Movies']
        data['IS_TS'] = data['Internet_Service'] + "_" + data['Tech_Support']
        
        return data.drop(columns=['Online_Security', 'Online_Backup', 'Device_Protection', 
                                   'Streaming_TV','Internet_Service', 'Streaming_Movies', 'Tech_Support'])

    def encode_categorical(self, data):
        """Robust categorical encoding."""
        columns = ['S_TV_M', 'IS_TS', 'O_B_S_Dp', 'Contract', 'Payment_Method']
        data_temp = data.copy()
        for col in columns:
            data_temp[col] = pd.Categorical(data_temp[col]).codes
        return data_temp

    def apply_kmodes(self, data_temp):
        """Apply KModes clustering with error handling."""
        if self.kmodes is None:
            raise ValueError("KModes model not loaded")
        return self.kmodes.predict(data_temp)

    def finalize_features(self, data):
        """Final feature preparation."""
        final_c = ['tenure', 'Monthly_Charges', 'gender_Female', 'gender_Male',
       'Senior_Citizen _No', 'Senior_Citizen _Yes', 'Is_Married_No',
       'Is_Married_Yes', 'Dependents_No', 'Dependents_Yes', 'Dual_No',
       'Dual_No phone service', 'Dual_Yes', 'Contract_Month-to-month',
       'Contract_One year', 'Contract_Two year', 'Paperless_Billing_No',
       'Paperless_Billing_Yes', 'Payment_Method_Bank transfer (automatic)',
       'Payment_Method_Credit card (automatic)',
       'Payment_Method_Electronic check', 'Payment_Method_Mailed check',
       'O_B_S_Dp_No internet service_No internet service_No internet service',
       'O_B_S_Dp_No_No_No', 'O_B_S_Dp_No_No_Yes', 'O_B_S_Dp_No_Yes_No',
       'O_B_S_Dp_No_Yes_Yes', 'O_B_S_Dp_Yes_No_No', 'O_B_S_Dp_Yes_No_Yes',
       'O_B_S_Dp_Yes_Yes_No', 'O_B_S_Dp_Yes_Yes_Yes',
       'S_TV_M_No internet service_No internet service', 'S_TV_M_No_No',
       'S_TV_M_No_Yes', 'S_TV_M_Yes_No', 'S_TV_M_Yes_Yes', 'IS_TS_DSL_No',
       'IS_TS_DSL_Yes', 'IS_TS_Fiber optic_No', 'IS_TS_Fiber optic_Yes',
       'IS_TS_No_No internet service', 'Cluster_0', 'Cluster_1', 'Cluster_2',
       'Cluster_3', 'Cluster_4']
        data['Cluster'] = data['Cluster'].astype(str)
        final_data = pd.DataFrame(columns=final_c)
        # Normalize numerical features
        for feature in self.numerical_features:
            scaler = MinMaxScaler()
            data[feature] = scaler.fit_transform(data[[feature]])
        
        # One-hot encode categorical features
        categorical_features = data.columns.drop(self.numerical_features )
        data = pd.get_dummies(data, columns=list(categorical_features))
        for i in final_c:
            if i in data.columns:
                final_data[i] = data[i]
            else:
                final_data[i] = False
        return final_data

    def predict_proba(self, X):
        """Predict churn probability."""
        return self.catboost_model.predict_proba(X)[:, 1]



class AdvancedPromptEngineer:
    """Advanced prompt engineering and context management."""

    @staticmethod
    def validate_context(context, required_fields):
        """Validate context and return missing fields."""
        missing_fields = [field for field in required_fields if field not in context]
        return missing_fields

    @staticmethod
    def classify_churn_risk(probability):
        """Classify churn risk based on probability."""
        if probability is None:
            return "Unknown"
        elif probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Moderate Risk"
        else:
            return "High Risk"

    @staticmethod
    def construct_contextual_prompt(context,user_input, churn_probability=None,):
        """Create a well-structured, context-aware prompt."""
        # Generate a human-readable summary of the context
        context_summary = "\n".join([
            f"{key.replace('_', ' ').title()}: {value}" for key, value in context.items()
        ])
        
        # Classify the churn risk based on the probability
        churn_risk = AdvancedPromptEngineer.classify_churn_risk(churn_probability)
        
        # Determine the additional instructions based on churn risk
        if churn_risk == 'Unknown':
            extra_prompt = (
                "Avoid disclosing the churn probability. Instead, ask the user for "
                "more client-related information to assist with the analysis."
            )
        else:
            extra_prompt = (
                f"Inform the user about the churn details. Mention that the churn risk is classified "
                f"as '{churn_risk}', and the probability is {churn_probability:.2f}."
            )
                    
        # Combine all components into the final prompt
        prompt = (
            "You are an intelligent and professional assistant tasked with helping the marketing team analyze and predict customer churn. "
            "Your role is to respond as a knowledgeable assistant. "
            "Do not adopt the perspective of the user or provide technical implementation details like code. Focus on being helpful, professional, and user-friendly.\n\n"
            f"Assistant Action Required if the user asked about churn:{extra_prompt}\n\n"
            f"User Query: {user_input}\n"
            "if the user does not ask about churn, respond normally to his query.\n"
            "As the assistant, respond with insightful and relevant information based on the context and query provided."
        )

        return prompt

class InputParser:
    @staticmethod
    def parse_input(user_input):
        """Enhanced input parsing with flexible matching."""
        mapping = {
            "gender": r"(male|female)",
            "Senior_Citizen": r"(yes|no|true|false)",
            "Is_Married": r"(yes|no|true|false)",
            "Dependents": r"(yes|no|true|false)",
            "tenure": r"\d+",
            "Dual": r"(no phone service|no|yes)",
            "Internet_Service": r"(dsl|fiber optic|no)",
            "Online_Security": r"(yes|no|no internet service)",
            "Online_Backup": r"(yes|no|no internet service)",
            "Device_Protection": r"(yes|no|no internet service)",            
            "Tech_Support": r"(yes|no|no internet service)",
            "Streaming_TV": r"(yes|no|no internet service)",
            "Streaming_Movies": r"(yes|no|no internet service)",
            "Contract": r"(month-to-month|one year|two year)",
            "Paperless_Billing": r"(yes|no|true|false)",
            "Payment_Method": r"(electronic check|mailed check|bank transfer \(automatic\)|credit card \(automatic\))",
            "Monthly_Charges": r"\d+(\.\d+)?"
        }

        data = {}
        for field, pattern in mapping.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                data[field] = match.group(0).lower()

        if data.get("Internet_Service") == "no":
            internet_service_fields = [
                "Online_Security", "Online_Backup", "Device_Protection", 
                "Streaming_TV", "Streaming_Movies", "Tech_Support"
            ]
            for field in internet_service_fields:
                data[field] = "no internet service"

        return data

class ChurnPredictionChatbot:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.prompt_engineer = AdvancedPromptEngineer()
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            self.model_name = os.getenv('INFERENCE_MODEL', 'meta-llama/Llama-3.2-1B')
            self.llm = InferenceClient(model=self.model_name, token=hf_token)
            logger.info(f"Initialized Hugging Face Inference Client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Critical Initialization Error: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize LLM: {e}")
        self.required_fields = [
            "gender", "Senior_Citizen", "Is_Married", "Dependents", "tenure",
            "Dual", "Contract", "Paperless_Billing", "Payment_Method", "Monthly_Charges",
            "Online_Security", "Online_Backup", "Device_Protection",
            "Streaming_TV", "Streaming_Movies", "Internet_Service", "Tech_Support"
        ]

    def preprocess_input(self, user_input):
        """Parse and preprocess user input."""
        flag = True
        try:
            logger.info(f"User input: {user_input}")
            parsed_input = InputParser.parse_input(user_input)
            missing_fields = self.prompt_engineer.validate_context(parsed_input, self.required_fields)
            if missing_fields:
                logger.warning(f"Missing fields in user input: {missing_fields}")
                #raise ValueError(f"Please provide values for the missing fields: {', '.join(missing_fields)}")
                flag =  False

            df = pd.DataFrame([parsed_input])
            logger.info("Input successfully parsed and converted to dataframe.")
            return df,flag
        except Exception as e:
            logger.error(f"Error in input preprocessing: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate_response(self, user_input):
        """Generate chatbot response based on user input."""
        try:
            # Preprocess input
            df,flag = self.preprocess_input(user_input)
            # Predict churn probability
            churn_probability = None
            if flag:
                processed_data = self.pipeline.preprocess_data(df)
                churn_probability = self.pipeline.predict_proba(processed_data)[0]
                logger.info(f"Predicted churn probability: {churn_probability}")

            # Construct prompt and generate response
            context = df.iloc[0].to_dict()
            prompt = self.prompt_engineer.construct_contextual_prompt(context,user_input, churn_probability)
            logger.info(f"Generated prompt: {prompt}")

            # Use LLM to generate a response
            llm_response = self.llm.text_generation(prompt, max_new_tokens=500, temperature=0.9,repetition_penalty=1.1)
            logger.info(f"LLM response: {llm_response}")
            return llm_response
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Could not generate response: {e}")

def main():
    """Main function to run the Churn Prediction Chatbot."""
    try:
        # Initialize the pipeline and chatbot
        pipeline = CatBoostPipeline(
            model_path=os.getenv('MODEL_PATH', 'cb_model.cbm'),
            kmodes_path=os.getenv('KMODES_PATH', 'k_model.pkl')
        )
        chatbot = ChurnPredictionChatbot(pipeline)

        user_input = ""
        response = ""

        @app.route('/api/v1/response', methods=['POST'])
        def getResponse():
            global user_input
            user_input = request.json['user_input']
            print(user_input)
            response = chatbot.generate_response(user_input)
            return jsonify({"response": response})

        app.run(host='0.0.0.0' , port=5000)
    except Exception as e:
        logger.error(f"Critical Error in main function: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
