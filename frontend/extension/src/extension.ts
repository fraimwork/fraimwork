import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { Graph } from 'graphlib';

const workspaceFolders = vscode.workspace.workspaceFolders;

if (workspaceFolders) {
    workspaceFolders.forEach((folder: { uri: { fsPath: any; }; }) => {
        console.log('Folder: ', folder.uri.fsPath);
    });
}

function getWebviewContent(context: vscode.ExtensionContext): string {
    const htmlFilePath = path.join(context.extensionPath, 'media', 'webview.html');
    const htmlContent = fs.readFileSync(htmlFilePath, 'utf8');
    return htmlContent;
}

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('fraimwork.examineDependencies', async () => {
        vscode.window.showInformationMessage('Analyzing dependencies...');

        const files = await getFiles();
        console.log('Files: ', files);
        const map: { [key: string]: string[] } = {};

        const promises = files.map(file => findDependencies(file));
        const results = await Promise.all(promises);
        // results.forEach((dependencies, i) => {
        //     const relativePath = path.relative(workspaceFolders![0].uri.fsPath, files[i].fsPath);
        //     map[relativePath] = Array.from(dependencies);
        // });
        results.forEach((dependencies, i) => {
            const relativePath = path.relative(workspaceFolders![0].uri.fsPath, files[i].fsPath);
            console.log(`Processing file: ${files[i].fsPath}`);  // Log the file being processed
            console.log(`Dependencies found: ${Array.from(dependencies)}`);  // Log the dependencies found
            map[relativePath] = Array.from(dependencies);
        });

        const graph = new Graph();
        // Add notes from map
        for (const filePath in map) {
            const fileName = path.basename(filePath);
            graph.setNode(filePath, { label: fileName });
        }
        // Add edges
        for (const filePath in map) {
            for (const dep of map[filePath]) {
                graph.setEdge(filePath, dep);
            }
        }

        visualizeDependencyGraph(graph);
    });

    context.subscriptions.push(disposable);
}

async function findDependencies(file: vscode.Uri): Promise<Set<string>> {
    const dependencies = new Set<string>();
    
    const document = await vscode.workspace.openTextDocument(file);
    const text = document.getText();
    const lines = text.split('\n');
    if (lines) {
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (!line.trim().startsWith('import') && !line.trim().startsWith('from')) {
                continue;
            }
            const tokens = line.split(' ');

            for (const token of tokens) {
                console.log(token);
                const location = new vscode.Position(i, line.indexOf(token) + token.length/2);

                // Get definition using the language server
                const locations = await vscode.commands.executeCommand<vscode.LocationLink[]>(
                    'vscode.executeDefinitionProvider',
                    file,
                    location
                );

                if (locations && locations.length > 0) {
                    try {
                        const dependencyFile = locations[0].targetUri.fsPath;
                        // Ensure the dependency file is not the current file and that it is in the same workspace
                        if (dependencyFile !== file.fsPath && workspaceFolders && workspaceFolders.some((folder) => dependencyFile.startsWith(folder.uri.fsPath))) {
                            const relativePath = path.relative(workspaceFolders[0].uri.fsPath, dependencyFile);
                            // Get symbols within dependency
                            const dependencyUri = vscode.Uri.file(dependencyFile);
                            if (dependencyUri) {
                                const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
                                    'vscode.executeDocumentSymbolProvider',
                                    dependencyUri
                                );
                                if (symbols) {
                                    for (const symbol of symbols) {
                                        if (symbol.kind == 5 || symbol.kind == 12 || symbol.kind == 1) {
                                            const symbolKind = SymbolKindMap[symbol.kind] || "Unknown";
                                            dependencies.add(`${relativePath}#${symbol.name} (${symbolKind})`);
                                        } else {
                                            dependencies.add(dependencyFile);
                                        }
                                    }
                                } 
                            }
                            
                        }
                    } catch (error) {
                        console.log("Error retrieving file " +  error);
                    }
                }
            }
        }
    }

    return dependencies;
}

async function getFiles(): Promise<vscode.Uri[]> {
    const pattern = '**/*.{js,jsx,ts,tsx,py,dart}';
    const ignoredPattern = '**/node_modules/**'; // Adjust this pattern based on the file types you're interested in
    const files = await vscode.workspace.findFiles(pattern, ignoredPattern);
    return files;
}

async function getSymbols(uri: vscode.Uri): Promise<vscode.DocumentSymbol[]> {
    const document = await vscode.workspace.openTextDocument(uri);
    const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>('vscode.executeDocumentSymbolProvider', document.uri);

    return symbols || [];
}

function visualizeDependencyGraph(dependencyGraph: Graph) {
    console.log('Visualizing Dependency Graph');
    const panel = vscode.window.createWebviewPanel(
        'dependencyGraph',
        'Dependency Graph',
        vscode.ViewColumn.One,
        {
            enableScripts: true  // Allow JavaScript execution in the webview
        }
    );

    const [nodes, edges] = [dependencyGraph.nodes(), dependencyGraph.edges()];
    const graphData = { nodes, edges };

    console.log('Graph Data:', graphData);  // Debug output

    panel.webview.html = renderGraph(graphData);
}

function renderGraph(graphData: { nodes: string[], edges: { v: string, w: string }[] }): string {
    return `
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {
                margin: 0;
                overflow: hidden;
            }
            svg {
                width: 100vw;
                height: 100vh;
            }
            .node {
                fill: #007ACC;
            }
            .link {
                stroke: #ccc;
                stroke-width: 2px;
            }
        </style>
    </head>
    <body>
        <svg></svg>
        <script>
            const nodes = ${JSON.stringify(graphData.nodes.map(node => ({ id: node })))};
            const links = ${JSON.stringify(graphData.edges.map(edge => ({
                source: edge.v, target: edge.w
            })))};

            const svg = d3.select('svg')
                .attr('width', window.innerWidth)
                .attr('height', window.innerHeight)
                .call(d3.zoom().on('zoom', (event) => {
                    svg.attr('transform', event.transform);  // Apply zoom/pan to SVG
                }))
                .append('g');  // Append group for zooming

            const width = window.innerWidth;
            const height = window.innerHeight;

            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2));

            const link = svg.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(links)
                .enter()
                .append('line')
                .attr('class', 'link');

            const node = svg.append('g')
                .attr('class', 'nodes')
                .selectAll('circle')
                .data(nodes)
                .enter()
                .append('circle')
                .attr('class', 'node')
                .attr('r', 10);

            const label = svg.append('g')
                .attr('class', 'labels')
                .selectAll('text')
                .data(nodes)
                .enter()
                .append('text')
                .attr('class', 'label')
                .text(d => d.id)
                .style('font-size', '12px');  // Adjust text size for better readability

            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                label
                    .attr('x', d => d.x + 12)
                    .attr('y', d => d.y + 3);
            });
        </script>
    </body>
    </html>
    `;
}

// This method is called when your extension is deactivated
export function deactivate() {}

const SymbolKindMap: { [key: number]: string } = {
    1: 'File',
    2: 'Module',
    3: 'Namespace',
    4: 'Package',
    5: 'Class',
    6: 'Method',
    7: 'Property',
    8: 'Field',
    9: 'Constructor',
    10: 'Enum',
    11: 'Interface',
    12: 'Function',
};