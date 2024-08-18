const fs = require('fs');
const path = require('path');
const { createGraph, alg } = require('graphlib');  // You can use graphlib for graph functionality

class FileTree {
    constructor() {
        this.graph = createGraph();
        this.root = '.';
    }

    getFiles() {
        return this.graph.nodes().filter(node => this.graph.node(node).content !== undefined);
    }

    getClosestFileName(fileName, editDistance) {
        const files = this.getFiles();
        let closestFile = files.reduce((prev, curr) => {
            return editDistance(curr, fileName) < editDistance(prev, fileName) ? curr : prev;
        });
        return closestFile;
    }

    rootNode() {
        return this.root;
    }

    copy(withDepth = 10 ** 6) {
        const newTree = new FileTree();
        const addSubtree = (node, depth) => {
            if (depth > withDepth) return;
            this.graph.successors(node).forEach(successor => {
                newTree.graph.setNode(successor, this.graph.node(successor));
                newTree.graph.setEdge(node, successor);
                addSubtree(successor, depth + 1);
            });
        };

        newTree.graph.setNode(this.rootNode(), this.graph.node(this.rootNode()));
        addSubtree(this.rootNode(), 1);
        newTree.root = this.root;
        return newTree;
    }

    subfiletree(node) {
        const descendants = alg.dijkstra(this.graph, node).map(d => d.node);
        const fileTree = new FileTree();
        descendants.forEach(descendant => {
            fileTree.graph.setNode(descendant, this.graph.node(descendant));
        });
        fileTree.root = node;
        return fileTree;
    }

    static fromDir(rootPath) {
        return buildFileTreeDAG(rootPath);
    }

    leafNodes() {
        return this.graph.nodes().filter(node => this.graph.successors(node).length === 0);
    }

    toString() {
        return filetreeToString(this);
    }

    toJSON() {
        return this.toString();
    }
}

function buildFileTreeDAG(rootPath) {
    const dag = new FileTree();
    const walkDir = (dirPath) => {
        const entries = fs.readdirSync(dirPath, { withFileTypes: true });
        entries.forEach(entry => {
            const fullPath = path.join(dirPath, entry.name);
            const relativePath = path.relative(rootPath, fullPath);

            if (entry.isDirectory()) {
                const parentDir = path.relative(rootPath, dirPath);
                dag.graph.setNode(relativePath, { name: entry.name, path: fullPath });
                dag.graph.setEdge(parentDir, relativePath);
                walkDir(fullPath);
            } else if (entry.isFile()) {
                const parentDir = path.relative(rootPath, dirPath);
                dag.graph.setNode(relativePath, { name: entry.name, path: fullPath });

                try {
                    const content = fs.readFileSync(fullPath, 'utf8');
                    dag.graph.node(relativePath).content = content.length ? content : "*EMPTY FILE*";
                } catch (err) {
                    dag.graph.node(relativePath).content = "*BINARY FILE*";
                }
                dag.graph.setEdge(parentDir, relativePath);
            }
        });
    };

    walkDir(rootPath);
    return dag;
}

function filetreeToString(tree) {
    const dfs = (node, indent = '') => {
        let result = '';
        const items = tree.graph.successors(node);
        items.forEach(item => {
            const itemAttr = tree.graph.node(item);
            const itemPath = itemAttr.path;

            if (fs.existsSync(itemPath) && fs.lstatSync(itemPath).isDirectory()) {
                result += `${indent}├── ${itemAttr.name}\\\n`;
                result += dfs(item, indent + '│   ');
            } else {
                result += `${indent}├── ${itemAttr.name}\n`;
            }
        });
        return result;
    };

    return dfs(tree.rootNode());
}

function writeFileTree(fileTreeString, baseDir) {
    const lines = fileTreeString.split('\n');
    const stack = [baseDir];

    lines.forEach(line => {
        const indentLevel = line.length - line.trimStart().length;
        const currentDir = stack[Math.floor(indentLevel / 4)];  // Assuming 4 spaces per indentation level
        const newPath = path.join(currentDir, line.split(' ').pop());

        if (line.endsWith('\\')) {
            fs.mkdirSync(newPath, { recursive: true });
            stack[Math.floor(indentLevel / 4) + 1] = newPath;
        } else {
            fs.mkdirSync(path.dirname(newPath), { recursive: true });
            fs.writeFileSync(newPath, '');
        }
    });
}

module.exports = { FileTree, buildFileTreeDAG, filetreeToString, writeFileTree };
