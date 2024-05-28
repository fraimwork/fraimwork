import React from 'react';

const Spacer = ({ size = null, axis = 'vertical' }) => {
    const styles = size ? {
        width: axis === 'horizontal' ? `${size}px` : '0',
        height: axis === 'vertical' ? `${size}px` : '0',
    } : {
        flex: 1, // Use flex to occupy available space
        display: 'flex', // Needed for flex to work
        ...(axis === 'horizontal' ? { flexDirection: 'row' } : {}), // For horizontal spacing
    };
    return <div style={styles} />;
};

export default Spacer;
