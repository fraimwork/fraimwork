// Import the axios library for making HTTP requests
import axios from 'axios';

// Define the base URL for the Flask backend API
const API_BASE_URL = "https://10.88.0.4:8080"; 

const handleError = (error) => {
  console.error('Error:', error);
  throw error;
};

export const translate = async (repo, target) => {
    console.log(`Calling API at ${API_BASE_URL}`)
    const response = await axios.get(`${API_BASE_URL}/test`).catch({handleError});
    console.log(`Response: ${response.data}`)
    return response.data;
}
