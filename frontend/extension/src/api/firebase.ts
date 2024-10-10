import { initializeApp } from "firebase/app";
import { getAuth, signInWithPopup, GithubAuthProvider, Auth } from "firebase/auth";

const firebaseConfig: { [key: string]: string } = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT_ID.appspot.com",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth: Auth = getAuth(app);

const provider: GithubAuthProvider = new GithubAuthProvider();

export { app, auth, provider };
