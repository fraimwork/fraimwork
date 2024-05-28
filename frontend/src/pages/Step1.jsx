import React, { useState } from "react";
import { Container } from "@mui/material";
import Dropdown from "./components/Dropdown";
import Button from "./components/Button";
import TextField from "./components/TextField";
// import { functions } from './firebase';
// import { httpsCallable } from "firebase/functions";

const frameworks = [
    { value: "react", label: "React" },
    { value: "vue", label: "Vue" },
    { value: "angular", label: "Angular" },
];

export function Home() {
    const [repoLink, setRepoLink] = useState('');
    const [targetFramework, setTargetFramework] = useState('react');
    
    const handleSubmit = async () => {
        // const translateCode = httpsCallable(functions, 'translateCode');
        // const result = await translateCode({ repoLink, targetFramework });
        // console.log(result.data.message);
        console.log('Submitted');
    };

    return (
        <Container style={{ justifyContent: 'center', alignItems: 'center' }}>
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <TextField
                    variant="outlined"
                    label="GitHub Repo Link"
                    value={repoLink}
                    onChange={(e) => setRepoLink(e.target.value)}
                    style={{ width: '300px', borderRadius: '50px' }}
                />
                <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleSubmit}
                    style={{ borderRadius: '50px' }}
                >
                    Next
                </Button>
            </form>
        </Container>
    );
}