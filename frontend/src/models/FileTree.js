// src/components/FileTree.js
import React, { useState } from 'react';
import styled from 'styled-components';
import TreeNode from './TreeNode';

const TreeContainer = styled.div`
    font-family: Arial, sans-serif;
`;

const FileTree = ({ data }) => {
    const [treeData, setTreeData] = useState(data);

    const moveNode = (draggedId, targetId) => {
        // Logic to move the node within the treeData
    };

    const renderTree = (nodes, level = 0) => {
        return nodes.map((node) => (
        <div key={node.id}>
            <TreeNode node={node} level={level} moveNode={moveNode} />
            {node.children && renderTree(node.children, level + 1)}
        </div>
        ));
    };

    return <TreeContainer>{renderTree(treeData)}</TreeContainer>;
};

export default FileTree;
