// src/App.jsx
import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { auth } from './firebase';
import SignInPage from './pages/SignInPage';
import HomePage from './pages/HomePage';

const App = () => {
    const [user, setUser] = useState(null);

    useEffect(() => {
        const unsubscribe = auth.onAuthStateChanged((user) => {
            setUser(user);
        });
        return () => unsubscribe();
    }, []);

    return (
        <Router>
            <Routes>
                <Route path="/signin" element={<SignInPage />}>
                </Route>
                <Route path="/" element={<HomePage />}>
                </Route>
            </Routes>
        </Router>
    );
};

export default App;