// src/models/TreeNode.js
import React from 'react';
import { useDrag, useDrop } from 'react-dnd';
import styled from 'styled-components';

const NodeContainer = styled.div`
    padding-left: ${(props) => props.level * 20}px;
    cursor: pointer;
`;

const TreeNode = ({ node, level, moveNode }) => {
    const [, drag] = useDrag({
        type: 'NODE',
        item: { id: node.id },
    });

    const [, drop] = useDrop({
        accept: 'NODE',
        drop: (item) => moveNode(item.id, node.id),
    });

    return (
        <NodeContainer ref={(instance) => drag(drop(instance))} level={level}>
        {node.type === 'folder' ? '📁' : '📄'} {node.name}
        </NodeContainer>
    );
};

export default TreeNode;
