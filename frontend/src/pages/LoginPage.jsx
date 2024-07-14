// src/Login.js

import React from "react";
import { auth, githubProvider } from "../firebase";

const Login = () => {
    const signInWithGithub = async () => {
        try {
            const result = await auth.signInWithPopup(githubProvider);
            const token = result.credential.accessToken;
            const user = result.user;

            // Fetch the GitHub PAT using the token
            const response = await fetch('https://api.github.com/user', {
                headers: {
                    Authorization: `token ${token}`,
                },
            });

            const data = await response.json();
            const pat = data.token;  // Assuming `data.token` contains the PAT

            // Send the PAT to your server to store it securely
            await fetch('/add-github-pat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    uid: user.uid,
                    pat: pat,
                }),
            });
        } catch (error) {
        console.error("Error signing in with GitHub", error);
        }
    };

    return (
        <div style={styles.container}>
            <h1 style={styles.header}>Login</h1>
            <button style={styles.button} onClick={signInWithGithub}>
                Sign in with GitHub
            </button>
        </div>
    );
};

const styles = {
    container: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        backgroundColor: '#f0f0f0',
    },
    header: {
        fontSize: '2rem',
        marginBottom: '2rem',
    },
    button: {
        padding: '0.5rem 1rem',
        fontSize: '1rem',
        backgroundColor: '#333',
        color: '#fff',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
    },
};

export default Login;
