import React from "react";
import { Button as MaterialButton } from "@mui/material";

const Button = ({ variant, color, onClick, style, children }) => {
    return (
        <MaterialButton variant={variant} color={color} onClick={onClick} style={style}>
        {children}
        </MaterialButton>
    );
}

export default Button;