// src/App.js
import React, { useState } from 'react';
import { Steps, Button, Typography, Input, Select } from 'antd';
import Spacer from './components/Spacer'; // Assuming you have a Spacer component

const { Step } = Steps;
const { Title } = Typography;
const { Option } = Select;

const steps = ['Enter Repo Link', 'Choose Target Framework', 'Visit Pull Request'];

const frameworks = [
    { value: 'react-native', label: 'React Native' },
    { value: 'flutter', label: 'Flutter' },
];

function App() {
    const [repoLink, setRepoLink] = useState('');
    const [targetFramework, setTargetFramework] = useState('react-native');
    const [activeStep, setActiveStep] = useState(0);

    const handleNext = () => {
        switch (activeStep) {
            case 0:
                // Contact the Gemini API to translate the repo link
                break;
            case 1:
                // Contact the Gemini API to translate the repo link
                break;
            case 2:
                // Contact the Gemini API to translate the repo link
                break;
            default:
                break;
        }
        setActiveStep(activeStep + 1);
    };

    const handleBack = () => {
        setActiveStep(activeStep - 1);
    };

    const handleRepoLinkChange = (e) => {
        setRepoLink(e.target.value);
    };

    const handleTargetFrameworkChange = (value) => {
        setTargetFramework(value);
    };

    const getContent = (step) => {
        switch (step) {
            case 0:
                return (
                    <>
                        <Title level={4}>Paste your GitHub repository link below:</Title>
                        <Input placeholder="https://github.com/username/repo" value={repoLink} onChange={handleRepoLinkChange} />
                    </>

                );
            case 1:
                return (
                    <>
                        <Title level={4}>Choose the target framework:</Title>
                        <Select value={targetFramework} onChange={handleTargetFrameworkChange}>
                            {frameworks.map((framework) => (
                                <Option key={framework.value} value={framework.value}>
                                    {framework.label}
                                </Option>
                            ))}
                        </Select>
                    </>
                );
            case 2:
                return (
                    <>
                        <Title level={4}>Translation complete! Visit your pull request:</Title>
                        {/* Replace with actual PR link later */}
                        <a href="#">View Pull Request</a> 
                    </>
                );
            default:
                return <Title level={4}>Unknown Step</Title>;
        }
    };

    return (
        <div style={{ width: '60%', margin: '50px auto' }}>
            <Steps current={activeStep}>
                {steps.map((label) => (
                    <Step key={label} title={label} />
                ))}
            </Steps>
            <Spacer height={20} />
            <div>{getContent(activeStep)}</div>
            <Spacer height={20} />
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                {activeStep > 0 && (
                    <Button onClick={handleBack}>Back</Button>
                )}
                {activeStep < steps.length - 1 && (
                    <Button type="primary" onClick={handleNext}>Next</Button>
                )}
                {activeStep === steps.length - 1 && (
                    <Button type="primary">Done</Button>
                )}
            </div>
        </div>
    );
}

export default App;
