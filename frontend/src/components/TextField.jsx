import { TextField as MaterialTextField } from '@mui/material';
import React from 'react';

const TextField = ({ label, value, onChange, style }) => {
    return (
        <MaterialTextField
            variant="outlined"
            label={label}
            value={value}
            onChange={onChange}
            style={style}
        />
    );
}

export default TextField;
