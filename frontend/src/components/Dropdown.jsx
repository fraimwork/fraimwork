import React from 'react';
import { TextField, MenuItem } from '@mui/material';

const Dropdown = ({ label, value, onChange, options, style }) => {
    return (
        <TextField
            select
            label={label}
            value={value}
            onChange={onChange}
            style={style}
        >
            {options.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                    {option.label}
                </MenuItem>
            ))}
        </TextField>
    );
}

export default Dropdown;