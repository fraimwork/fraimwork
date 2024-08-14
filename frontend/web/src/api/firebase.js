// src/firebase.js
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth, GithubAuthProvider  } from "firebase/auth";
import { getFunctions, httpsCallable } from "firebase/functions";
import { getAnalytics } from "firebase/analytics";
import firebaseConfig from "firebaseConfig.json";

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const functions = getFunctions(app);
const analytics = getAnalytics(app);
const auth = getAuth(app);
const githubProvider = new GithubAuthProvider();
const signOut = auth.signOut;

export { db, functions, auth, githubProvider, analytics, signOut };