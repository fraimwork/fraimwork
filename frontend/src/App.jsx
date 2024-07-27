// src/App.jsx
import React, { useState } from 'react';
import LogoLight from './assets/logos/logo-light.png';
import LogoDark from './assets/logos/logo-dark.png'
import { Steps, List, Typography, Input, Select } from 'antd';
import { Button, Card } from 'ui-neumorphism'
import 'ui-neumorphism/dist/index.css'
import Spacer from './components/Spacer';
import { translate } from './models/BackendModel';

const { Step } = Steps;
const { Title } = Typography;
const { Option } = Select;

const steps = ['Enter Repo Link', 'Select Current Framework', 'Choose Target Framework', 'Visit Pull Request'];

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

    const handleMidTransitionFailure = (error) => {
        console.error(error);
        setActiveStep(activeStep - 1);
        setIsLoading(false);
    }

    const handleNext = async () => {
        setIsLoading(true);
        switch (activeStep) {
            case 0:
                // Retrieve the GitHub repo name
                break;
            case 1:
                // Contact the Gemini API to translate the repo link
                break;
            case 2:
                const link = await translate(repoLink, targetFramework).catch(handleMidTransitionFailure);
                setRepoLink(link);  
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
                        <Title level={4}>Select your current framework:</Title>
                        <Select value={sourceFramework} onChange={setSourceFramework}>
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
                        <Title level={4}>Choose your target framework:</Title>
                        <Select value={targetFramework} onChange={setTargetFramework}>
                            {frameworks.flatMap((framework) => framework.value === sourceFramework ? [] : (
                                <Option key={framework.value} value={framework.value}>
                                    {framework.label}
                                </Option>
                                )
                            )}
                        </Select>
                    </>
                );
            case 3:
                if (!repoLink) {
                    return (
                        <>
                            <Title level={4}>Error translating repository</Title>
                            <p>There was an error translating the repository. Please try again.</p>
                        </>
                    );
                }
                return (
                    <>
                        <Title level={4}>Translation complete! Visit your pull request:</Title>
                        {/* Replace with actual PR link later */}
                        <a href="www.google.com">View Pull Request</a> 
                    </>
                );
            default:
                return <Title level={4}>Unknown Step</Title>;
        }
    };

    return (
        <div style={{ width: '60%', margin: '50px auto' }}>
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}> {/* Centering container */}
                <img src={LogoDark} alt="App Logo" style={{ width: '150px', borderRadius: '30px' }} /> {/* Add the logo */}
            </div>
            <Card inset
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
                <div style={{ padding: '50px 0' }}>{getContent(activeStep)}</div> {/* Add padding to the content */}
                <Spacer height={20} />
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    {activeStep > 0 && (
                        <Button onClick={handleBack}>Back</Button>
                    )}
                    {activeStep < steps.length - 1 && (
                        <Button type="primary" onClick={handleNext}  disabled={isLoading || !repoLink} loading={isLoading}>Next</Button>
                    )}
                    {activeStep === steps.length - 1 && (
                        <Button>Done</Button>
                    )}
                </div>
            </Card>
        </div>
    );
}

export default App;
