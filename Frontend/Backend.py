from flask import Flask, request, jsonify
import requests
import pandas as pd

app = Flask(__name__)

# Function to send data to the Vertex AI endpoint
def send_data_to_vertex_ai(project, endpoint_id, location, instances):
    endpoint_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/{endpoint_id}:predict"
    headers = {
        "Authorization": f"Bearer {YOUR_ACCESS_TOKEN}"  # Replace YOUR_ACCESS_TOKEN with your access token
    }
    payload = {
        "instances": instances
    }
    response = requests.post(endpoint_url, headers=headers, json=payload)
    if response.status_code == 200:
        prediction_results = response.json()["predictions"]
        return prediction_results
    else:
        return "Failed to send data to Vertex AI endpoint"

# Define route to handle CSV file upload
@app.route('/process_csv', methods=['POST'])
def process_csv_file():
    # Get CSV file from request
    csv_file = request.files['file']
    
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert DataFrame to instances format expected by Vertex AI endpoint
    instances = df.to_dict(orient='records')
    
    # Send CSV data to Vertex AI endpoint
    prediction_results = send_data_to_vertex_ai("skilful-alpha-415221", "7340064749125632000", "us-central1", instances)
    
    # Add prediction results to DataFrame
    df_with_predictions = pd.DataFrame(prediction_results)
    
    # Save DataFrame with prediction results as CSV file
    output_csv_file = 'prediction_result.csv'
    df_with_predictions.to_csv(output_csv_file, index=False)
    
    # Filter fraudulent transactions
    fraudulent_transactions = df_with_predictions[df_with_predictions['prediction'] == 1]
    
    # Save fraudulent transactions as CSV file
    fraudulent_transactions_csv_file = 'fraudulent_transactions.csv'
    fraudulent_transactions.to_csv(fraudulent_transactions_csv_file, index=False)
    
    # Return JSON response with download links
    response_data = {
        'entire_csv_download_link': f'/download/{output_csv_file}',
        'fraudulent_transactions_download_link': f'/download/{fraudulent_transactions_csv_file}'
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
