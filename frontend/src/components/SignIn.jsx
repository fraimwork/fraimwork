// src/SignIn.js
import React from 'react';
import { auth, githubProvider } from '../firebase';

const SignIn = () => {
    const signInWithGitHub = () => {
        auth.signInWithPopup(githubProvider)
        .then((result) => {
            console.log(result);
        })
        .catch((error) => {
            console.error("Error signing in with GitHub: ", error);
        });
    };

    return (
        <div>
            <button onClick={signInWithGitHub}>Sign in with GitHub</button>
        </div>
    );
};

export default SignIn;