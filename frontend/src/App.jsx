// src/App.js
import React, { useState } from 'react';
import { Container, Typography, Stepper, Step, StepLabel } from '@mui/material';
import Dropdown from './components/Dropdown';
import Button from './components/Button';
import TextField from './components/TextField';
import Spacer from './components/Spacer';
// import { functions } from './firebase';
// import { httpsCallable } from "firebase/functions";

const steps = ['Step 1', 'Step 2', 'Step 3'];

const frameworks = [
    { value: 'react', label: 'React' },
    { value: 'vue', label: 'Vue' },
    { value: 'angular', label: 'Angular' },
];

function App() {
    const [repoLink, setRepoLink] = useState('');
    const [targetFramework, setTargetFramework] = useState('react');

    const [activeStep, setActiveStep] = React.useState(0);

    const handleNext = () => {
        setActiveStep(activeStep + 1);
    };

    const handleBack = () => {
        setActiveStep(activeStep - 1);
    };

    const handleReset = () => {
        setActiveStep(0);
    };

    const getContent = (step) => {
        switch (step) {
        case 0:
            return <Typography>Content for Step 1</Typography>;
        case 1:
            return <Typography>Content for Step 2</Typography>;
        case 2:
            return <Typography>Content for Step 3</Typography>;
        default:
            return <Typography>Unknown Step</Typography>;
        }
    };

    return (
        <Container style={{ justifyContent: 'center', alignItems: 'center', height: '1000vh' }}>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
                {getContent(activeStep)}
                <Button variant="contained" disabled={activeStep === 0} onClick={handleBack}>
                Back
                </Button>
                <Button variant="contained" onClick={handleNext}>
                {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                </Button>
                <Button variant="outlined" onClick={handleReset}>
                Reset
                </Button>
            </div>
            <Spacer>
                size={20}
            </Spacer>
            <div style={{ height: '2rem' }} />
            <Stepper activeStep={activeStep} alternativeLabel>
                {steps.map((label, index) => (
                <Step key={label}>
                    <StepLabel>{label}</StepLabel>
                </Step>
                ))}
            </Stepper>
        </Container>
    );
    
    const handleSubmit = async () => {
        // const translateCode = httpsCallable(functions, 'translateCode');
        // const result = await translateCode({ repoLink, targetFramework });
        // console.log(result.data.message);
        console.log('Submitted');
    };

    return (
        <Container style={{ justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <TextField
                    variant="outlined"
                    label="GitHub Repo Link"
                    value={repoLink}
                    onChange={(e) => setRepoLink(e.target.value)}
                    style={{ width: '300px', borderRadius: '50px' }}
                />
                <Dropdown>
                    label="Target Framework"
                    value={targetFramework}
                    onChange={(e) => setTargetFramework(e.target.value)}
                    options={frameworks}
                    style={{ width: '300px', borderRadius: '50px' }}
                </Dropdown>
                <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleSubmit}
                    style={{ borderRadius: '50px' }}
                >
                    Submit
                </Button>
            </form>
        </Container>
    );
}

export default App;