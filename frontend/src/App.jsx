// src/App.jsx
import React, { useState } from 'react';
import LogoLight from './assets/logos/logo-light.png';
import LogoDark from './assets/logos/logo-dark.png'
import { Steps, Button, List, Typography, Input, Select, Card } from 'antd';
import Spacer from './components/Spacer';
import { translate } from './models/BackendModel';

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
    const [sourceFramework, setSourceFramework] = useState('flutter');
    const [isLoading, setIsLoading] = useState(false);
    const [activeStep, setActiveStep] = useState(0);

    const handleNext = () => {
        setIsLoading(true);
        switch (activeStep) {
            case 0:
                // Retrieve the GitHub repo name
                break;
            case 1:
                // Contact the Gemini API to translate the repo link
                translate(repoLink, targetFramework).then((data) => {
                    console.log(data);
                }); 
                break;
            case 2:
                // Contact the Gemini API to translate the repo link
                break;
            default:
                break;
        }
        setActiveStep(activeStep + 1);
        setIsLoading(false);
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
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}> {/* Centering container */}
                <img src={LogoLight} alt="App Logo" style={{ width: '150px' }} /> {/* Add the logo */}
            </div>
            <Card 
                style={{ 
                    backgroundColor: 'rgba(220, 255, 255, 0.7)', // Adjust transparency as needed
                    backdropFilter: 'blur(10px)', // Adjust blur intensity
                    borderRadius: '10px',
                    padding: '20px'
                }}
            >
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
                        <Button type="primary" onClick={handleNext}  disabled={isLoading} loading={isLoading}>Next</Button>
                    )}
                    {activeStep === steps.length - 1 && (
                        <Button type="primary">Done</Button>
                    )}
                </div>
            </Card>
        </div>
    );
}

export default App;
