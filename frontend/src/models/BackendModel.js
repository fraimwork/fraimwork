// Import the axios library for making HTTP requests
import axios from 'axios';

// Define the base URL for the Flask backend API
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL; 

const handleError = (error) => {
  console.error('Error:', error);
  throw error;
};

export const translate = async (repo, target) => {
    const response = await axios.get(`${API_BASE_URL}/translate`, {
        params: {
            repo,
            target
        }
    }).catch((err) => handleError(err));
    return response.data;
}
