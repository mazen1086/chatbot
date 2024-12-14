# Churn Prediction Chatbot

## Overview

The **Churn Prediction Chatbot** is a streamlined system designed to predict customer churn using detailed customer data such as demographic information (e.g., age, gender, and marital status), service usage patterns (e.g., tenure, internet service type), and billing details (e.g., payment method and monthly charges). This system is particularly well-suited for subscription-based services and industries like telecommunications, where identifying at-risk customers is critical to improving retention and reducing churn. using a machine learning pipeline and provide insights through a conversational chatbot interface. It utilizes **Streamlit** for a frontend user interface, **Flask** for the backend, and integrates **CatBoost** and **KModes** models for predictions. Additionally, the chatbot leverages advanced prompt engineering and a language model for intelligent responses.

---

## Features

- **Predict Customer Churn**: Uses CatBoostClassifier to estimate churn probabilities based on customer data. CatBoostClassifier is chosen for its ability to handle categorical features efficiently without extensive preprocessing, and KModes clustering is used to identify distinct customer segments based on patterns in the data, providing actionable insights. based on customer data.
- **Advanced Prompt Engineering**: Dynamically generates insights based on customer data and churn probability.
- **Streamlit Interface**: Provides a simple UI for users to input data and view predictions.
- **Flask API**: Backend API to process user inputs and generate responses.
- **Logging**: Comprehensive logs for debugging and monitoring.

---

## Project Architecture

1. **Frontend**: Streamlit-based UI for user interaction.
2. **Backend**: Flask API that handles data processing and response generation.
3. **Machine Learning**:
   - **CatBoostClassifier**: Predicts churn probability.
   - **KModes Clustering**: Groups customers into clusters.
4. **Language Model**: Uses a HuggingFace-based model for response generation.

---

## Installation

### Prerequisites

- Python 3.8+
- Recommended system configuration: At least 8GB RAM and a multi-core CPU for efficient processing.
- Supported OS: Windows, Linux, or macOS.
- Required libraries:
  ```bash
  pip install streamlit flask pandas joblib catboost kmodes scikit-learn dotenv huggingface_hub
  ```
- Pre-trained models:
  - CatBoostClassifier model
  - KModes clustering model

### Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/mazen1086/chatbot.git
   cd chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with the following keys:

   ```plaintext
   MODEL_PATH=<path_to_catboost_model>
   KMODES_PATH=<path_to_kmodes_model>
   HUGGINGFACE_TOKEN=<your_huggingface_api_token>
   INFERENCE_MODEL=<huggingface_model_name>
   ```

4. Run the Flask backend:

   ```bash
   python bot.py
   ```

5. Run the Streamlit frontend:

   ```bash
   streamlit run front.py
   ```

---

## Usage

1. Open the Streamlit UI in your browser.
2. Enter customer details in the input box.
3. Click on "Predict and Generate Response".
4. View the predicted churn probability and chatbot-generated insights.

### Using the API with an External Frontend

To integrate the Flask API with an external frontend, follow these steps:

1. **API Endpoint**:

   - URL: `http://<host>:5000/api/v1/response`
   - Method: `POST`
   - Payload: JSON object containing a single key `user_input` with customer details as its value.
     Example:
     ```json
     {
       "user_input": "gender: male, tenure: 12, Monthly_Charges: 50.5"
     }
     ```

2. **Request Headers**:

   - Content-Type: `application/json`

3. **Response**:

   - A JSON object containing the generated response from the chatbot.
     Example:
     ```json
     {
       "response": "The customer has a high churn risk with a probability of 0.85."
     }
     ```

4. **Integration Examples**:

   - **JavaScript Fetch**:

     ```javascript
     const data = {
         user_input: "gender: male, tenure: 12, Monthly_Charges: 50.5"
     };

     fetch('http://<host>:5000/api/v1/response', {
         method: 'POST',
         headers: {
             'Content-Type': 'application/json'
         },
         body: JSON.stringify(data)
     })
     .then(response => response.json())
     .then(data => console.log('Response:', data.response))
     .catch(error => console.error('Error:', error));
     ```

   - **Python Request**:

     ````python
     import requests

     url = 'http://<host>:5000/api/v1/response'
     payload = {
         "user_input": "gender: male, tenure: 12, Monthly_Charges: 50.5"
     }
     headers = {
         'Content-Type': 'application/json'
     }

     response = requests.post(url, json=payload, headers=headers)
     if response.status_code == 200:
         print('Response:', response.json().get('response'))
     else:
         print('Error:', response.status_code, response.text)
     ````

5. Enter customer details in the input box.

6. Click on "Predict and Generate Response".

7. View the predicted churn probability and chatbot-generated insights.

---

## Code Pipeline

The following diagram illustrates the sequence and pipeline of the code:

Error handling and logging are implemented at each step:

- **Input Parsing & Validation**: Logs missing or invalid fields and raises detailed errors for user input corrections.
- **Feature Engineering & Encoding**: Captures and logs any discrepancies in the data schema or processing steps.
- **Churn Prediction**: Records model predictions, including probabilities and anomalies.
- **Customer Clustering**: Monitors the KModes clustering process for errors or unexpected outputs.
- **Prompt Generation**: Logs the prompts constructed for the language model, including any fallback mechanisms for incomplete data.
- **LLM Response Generation**: Tracks API requests to the language model and handles timeouts or response errors.

```mermaid
graph TD
A[User Input in Streamlit] -->|Send JSON| B[Flask API]
B --> C[Input Parsing & Validation]
C --> D[Feature Engineering & Encoding]
D --> E[Churn Prediction (CatBoost)]
D --> F[Customer Clustering (KModes)]
E --> G[Churn Risk Classification]
F --> G
G --> H[Generate Prompt]
H --> I[Generate Response using LLM]
I --> J[Return Response to Streamlit]
```

---

## Key Files

### Frontend

- **front.py**: Implements the Streamlit user interface.

### Backend

- **bot.py**: Handles Flask API logic, including preprocessing, churn prediction, and response generation.

### Models

- **CatBoost Pipeline**: Implements preprocessing, feature engineering, and churn prediction.
- **KModes**: Performs customer clustering.

---

## Logging and Monitoring

Logs are saved in a `logs/` directory and include:

- User input
- Preprocessing steps
- Prediction results
- API request and response traces

---

## Images
### Graph
![Execution flowchart](/images/execution_flowchart.png)

- I am using model `Llama-3.2-1b` which is a very small large language model

- The output of the chatbot is limited by the LLM capabiities 

### Images of complete data
- Prompt with comlete required data
![](/images/1.jpeg)
- The model parses the data correctly and outputs a churn probability
![](/images/3.jpeg)
![](/images/4.jpeg)
- for a better understanding of the chatbot's behaviour, please take a look on the logs directory
### Images of incomplete data
- prompt with incomplete data 
![](/images/5.jpeg)
- the chatbot's output is not relevant to the prompt by user but this is due to the LLM model's limitations
![](/images/6.jpeg)
- the chatbot doesn't output a probability with the incomplete data
![](/images/7.jpeg)

---

## Future Enhancements

- Deploy the project using Docker for seamless scalability.
- Improve the chatbot by fine-tuning the language model.
- Deploy it on cloud and use a better LLM 

---

## Acknowledgments

This project leverages:

- [Streamlit](https://streamlit.io)
- [Flask](https://flask.palletsprojects.com/)
- [HuggingFace](https://huggingface.co)
- [CatBoost](https://catboost.ai)
- [KModes](https://pypi.org/project/kmodes/)

---

## License

This project is licensed under the MIT License.

