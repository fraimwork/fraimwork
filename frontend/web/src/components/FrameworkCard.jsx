// src/components/FrameworkCard.jsx
import React from 'react';
import { Card } from 'antd';

const FrameworkCard = ({ framework }) => {
    return (
        <Card
        className="framework-card"
        cover={<img alt={framework.value} src={`../icons/${framework.value}.png`} />}
        >
        <div className="framework-card-name">{framework.label}</div>
        </Card>
    );
};

export default FrameworkCard;