// Import the axios library for making HTTP requests
import axios from 'axios';

// Define the base URL for the Flask backend API
const API_BASE_URL = "http://127.0.0.1:8080"; 

const handleError = (error) => {
    console.error('Error:', error);
    throw error;
};

export const translate = async (repo, source, target) => {
    console.log(`Calling API at ${API_BASE_URL}`);
    // Call the Flask backend API to translate the repo
    const data = {
        repo: repo,
        source: source,
        target: target,
    };
    const response = await axios.post(`${API_BASE_URL}/translate`, data).catch(handleError);
    console.log(`Response: ${response.data}`);
    return response.data;
}
