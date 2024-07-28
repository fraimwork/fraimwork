// src/pages/SignInPage.jsx

import React from "react";
import LogoLight from '../assets/logos/logo-light.png';
import LogoDark from '../assets/logos/logo-dark.png'
import { auth, githubProvider } from "../firebase";
import { Button, Card } from 'ui-neumorphism'
import 'ui-neumorphism/dist/index.css'

const SignInPage = () => {
    const signInWithGithub = async () => {
        try {
            const result = await auth.signInWithGithub(githubProvider);
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
        <div style={{ width: '60%', margin: '50px auto', display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center', }}>
            <Card inset
            style={{ 
                backgroundColor: 'rgba(220, 255, 255, 0.7)', // Adjust transparency as needed
                backdropFilter: 'blur(10px)', // Adjust blur intensity
                borderRadius: '10px',
                padding: '90px'
            }}
            >
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}> {/* Centering container */}
                        <img src={LogoDark} alt="App Logo" style={{ width: '150px', borderRadius: '30px' }} /> {/* Add the logo */}
                    </div>
                    <button style={styles.button} onClick={signInWithGithub}>
                        Sign in with GitHub
                    </button>
            </Card>
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

export default SignInPage;
